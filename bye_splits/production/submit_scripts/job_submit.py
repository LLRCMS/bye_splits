#!/usr/bin/env python

import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "../")
sys.path.insert(0, parent_dir)

from bye_splits.utils import params, common, job_helpers

from datetime import datetime
import subprocess
import yaml

class JobBatches:
    """Class for setting up job batches and setting configuration
    variables. The function setup_batches() will take the list in
    config[arguments[<iterOver>]] and return a list of lists containing
    <args_per_batch> values in each sublist. Example for five total values
    with <args_per_batch> = 2:
    [0.01, 0.02, 0.03, 0.04, 0.05] --> [[0.01, 0.02], [0.03, 0.04], [0.05]]"""

    def __init__(self, particle, config):
        self.particle = particle
        self.config   = config
        self.iterOver = config["job"]["iterOver"]
        self.particle_var = lambda part, var: config["job"][part][var]

    def setup_batches(self):
        total_vals = self.config["job"]["arguments"][self.iterOver]

        vals_per_batch = self.particle_var(self.particle, "args_per_batch")    

        batches = [total_vals[i: i + vals_per_batch] for i in range(0, len(total_vals), vals_per_batch)]
        
        return batches
    
class CondJobBase:
    def __init__(self, particle, config):
        self.particle     = particle
        self.script       = config["job"]["script"]
        self.iterOver     = config["job"]["iterOver"]
        self.args         = config["job"]["arguments"]
        self.queue        = config["job"]["queue"]
        self.proxy        = config["job"]["proxy"]
        self.local        = config["job"]["local"]
        self.user         = config["job"]["user"]
        self.batch        = JobBatches(particle, config)
        self.particle_dir = self.batch.particle_var(self.particle, "submit_dir")
        self.batches      = self.batch.setup_batches()

    def _check_script_arguments(self):
        script_args = subprocess.run([self.script, "--args"], capture_output=True, text=True)
        if script_args.returncode != 0:
            result = script_args.stdout.strip().split("\n")
        else:
            result = []
        
        assert(result == list(self.args.keys())), "Provided arguments {} do not match accepted arguments {}.".format(list(self.args.keys()), result)


    def _write_arg_values(self, current_version):
        """Adds the argument values, where the batch lists are converted
        to strings as [val_1, val_2, ...] --> "[val_1;val_2]".
        The choice of a semicolon as the delimiter is arbitrary but it
        cannot be a comma because this is the delimeter condor itself uses.

        Example:

        queue radius, particle from (
        [0.01, 0.02], photon
        )
        incorrectly assigns radius="[0.01", particle="0.02]"

        queue radius, particle from (
            [0.01;0.02], photon
        )
        correctly assigns radius="[0.01, 0.02]", particle="photon"
        """
        
        if "particles" in self.args.keys(): self.args["particles"] = self.particle
        arg_keys = ", ".join(self.args.keys())
        arg_keys = "queue " + arg_keys + " from (\n"

        current_version.append(arg_keys)

        sub_args = list(self.args.keys())
        sub_args.remove(self.iterOver)
        arg_vals = [self.args[key] for key in sub_args]

        for batch in self.batches:
            all_vals = [str(batch).replace(", ", ";")]+arg_vals
            all_vals = ", ".join(map(str, all_vals)) + "\n"
            current_version.append(all_vals)

        current_version.append(")")

    def write_exec_file(self):
        """"""
        script_dir, script_name = os.path.split(self.script)
        basename, ext = os.path.splitext(script_name)
        
        sub_dir = "{}subs/".format(self.particle_dir)

        common.create_dir(sub_dir)

        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        exec_file_name_template = "{}{}_exec.sh".format(sub_dir, script_basename)
        exec_file_versions = job_helpers.grab_most_recent(exec_file_name_template, return_all=True)

        current_version = []
        current_version.append("#!/usr/bin/env bash\n")
        current_version.append("export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch\n")
        current_version.append("export SITECONFIG_PATH=$VO_CMS_SW_DIR/SITECONF/T2_FR_GRIF_LLR/GRIF-LLR/\n")
        current_version.append("source $VO_CMS_SW_DIR/cmsset_default.sh\n")
        
        sub_args = list(self.args.keys())
        if self.iterOver is not None:
            sub_args.remove(self.iterOver)

            current_version.append('list=$1\n')
            current_version.append("cleaned_list=$(echo $list | tr -d '[]' | tr ';' '\n')\n")
            current_version.append('while IFS=";" read -r val; do\n')
            
            python_call = '    python {} --{} "$val"'.format(self.script, self.iterOver)
            for arg in sub_args:
                python_call +=' --{} {}'.format(arg, self.args[arg])
            python_call += "\n"

            current_version.append(python_call)
            current_version.append('done <<< "$cleaned_list"')

        else:
            python_call = 'python {}'.format(self.script)

            for arg in sub_args:
                python_call +=' --{} {}'.format(arg, self.args[arg])   

            current_version.append(python_call)

        self.sub_file = job_helpers.conditional_write(exec_file_versions, exec_file_name_template, current_version)

    def write_sub_file(self):
        """Writes the .sub script that is submitted to HT Condor.
        Follows the same naming convention and conditional_write()
        procedure as the previous function."""

        log_dir = "{}logs/".format(self.particle_dir)

        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        job_file_name_template = "{}jobs/{}.sub".format(self.particle_dir, script_basename)

        job_file_versions = job_helpers.grab_most_recent(job_file_name_template, return_all=True)

        current_version = []
        current_version.append("executable = {}\n".format(self.sub_file))
        current_version.append("Universe              = vanilla\n")
        
        #self._check_script_arguments()

        """Add argument keys, requring that the first argument corresponds to <iterOver>.
        This is because the handling of the iterated argument is handled differently by
        than the static variables by write_exec_file() which expectes <iterOver> to be
        passed first."""
        arg_keys = "Arguments ="
        sub_args = list(self.args.keys())
        if self.iterOver is not None:
            arg_keys += " $({})".format(self.iterOver)
            sub_args.remove(self.iterOver)
        for arg in sub_args:
            arg_keys += " $({})".format(arg)
        arg_keys += "\n"

        current_version.append(arg_keys)          

        current_version.append("output = {}{}_C$(Cluster)P$(Process).out\n".format(log_dir, script_basename))
        current_version.append("error = {}{}_C$(Cluster)P$(Process).err\n".format(log_dir, script_basename))
        current_version.append("log = {}{}_C$(Cluster)P$(Process).log\n".format(log_dir, script_basename))
        current_version.append("getenv                = true\n")
        current_version.append("T3Queue = {}\n".format(self.queue))
        current_version.append("WNTag                 = el7\n")
        current_version.append('+SingularityCmd       = ""\n')
        current_version.append("include: /opt/exp_soft/cms/t3/t3queue |\n")
        
        if len(self.args.keys()) > 0:
            self._write_arg_values(current_version)

        # Write the file only if an identical file doesn't already exist
        self.submission_file = job_helpers.conditional_write(job_file_versions, job_file_name_template, current_version) # Save to launch later

class CondJob:
    """Creates the job directories and files
    with prepare_jobs() and runs the jobs with
    launch_jobs()."""

    def __init__(self, particle, config):
        self.base = CondJobBase(particle=particle, config=config)

    def prepare_jobs(self):

        config_dir = self.base.particle_dir + "configs"
        job_dir    = self.base.particle_dir + "jobs"
        log_dir    = self.base.particle_dir + "logs"

        for d in (config_dir, job_dir, log_dir):
            common.create_dir(d)

        self.base.write_exec_file()
        self.base.write_sub_file()

    def launch_jobs(self):

        if self.base.local:
            machine = "local"
        else:
            machine = "llrt3.in2p3.fr"

        sub_comm = ["condor_submit"]

        if not self.base.local:
            print(
                "\nSending {} jobs on {}".format(self.base.particle, self.base.queue + "@{}".format(machine))
            )
            print("===============")
            print("\n")

        sub_args = []

        sub_args.append(self.base.submission_file)

        if self.base.local:
            comm = sub_args
        else:
            comm = sub_comm + sub_args

        print(str(datetime.now()), " ".join(comm))
        subprocess.run(comm)

if __name__ == "__main__":    
    with open(params.CfgPath, "r") as afile:
        config = yaml.safe_load(afile)

    job = CondJob("photons", config)
    job.prepare_jobs()
    job.launch_jobs()

    '''for particle in ("photons", "electrons", "pions"):
        job = CondJob(particle, config)
        job.prepare_jobs()
        job.launch_jobs()'''