#!/usr/bin/env python

import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "../")
sys.path.insert(0, parent_dir)

from bye_splits.utils import params, common, job_helpers
from bye_splits.utils.job_helpers import Arguments

from datetime import datetime
import re
import subprocess
import argparse
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
        self.script       = config["job"]["script"]
        self.queue        = config["job"]["queue"]
        self.proxy        = config["job"]["proxy"]
        self.local        = config["job"]["local"]
        self.user         = config["job"]["user"]

        self.true_args    = Arguments(self.script)
        self.default_args = {}
        for arg, val in config["job"]["arguments"].items():
            self.default_args["--"+arg] = val
        self.combined_args = self.true_args.verify_args(self.default_args)

        if "--particles" in self.combined_args:
            self.particle = self.combined_args["--particles"]
        else:
            self.particle     = particle

        self.batch        = JobBatches(particle, config)
        self.particle_dir = self.batch.particle_var(self.particle, "submit_dir")
        self.iterOver     = "--"+config["job"]["iterOver"]
        self.batches      = self.batch.setup_batches()

    def _get_condor_args(self):
        condor_args   = []
        for arg in self.combined_args:
            if "action" in self.true_args.accepted_args[arg] and self.true_args.accepted_args[arg]["action"]=="store_true":
                continue
            else:
                condor_args.append(arg.replace("--", ""))
        
        return condor_args

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
        
        condor_args = self._get_condor_args()

        arg_keys = "queue " + ", ".join(condor_args) + " from (\n"
        current_version.append(arg_keys)
        
        batch_strs = [str(batch).replace(", ", ";") for batch in self.batches]
        
        for batch in batch_strs:
            inner_test = []
            for key, val in self.combined_args.items():
                if "action" in self.true_args.accepted_args[key] and self.true_args.accepted_args[key]["action"]=="store_true":
                    continue
                else:
                    if key != self.iterOver:
                        inner_test.append(val)
                    else:
                        inner_test.append(batch)
            inner_test = ", ".join(map(str, inner_test)) + "\n"
            current_version.append(inner_test)
        
        current_version.append(")")

    def write_exec_file(self):
        """Writes the .sh file that the condor .sub file runs as the
        <executable>. This constitutes a few common exports and sourcing,
        and follows by writing the Python call to <script>, passing the given
        arguments and values. The contents of this file is written to a buffer
        list first, and conditional_write() will check the contents of other scripts
        in the same directory that contain the same basename. If an identical script
        already exists, will use this script. Otherwise, it will increment the version
        number (<script_name> + _v<#>). If no versions exist, the v0 version will be written."""

        script_dir, script_name = os.path.split(self.script)
        basename, ext = os.path.splitext(script_name)
        
        sub_dir = os.path.join(self.particle_dir, "subs/")

        common.create_dir(sub_dir)

        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        exec_file_name_template = "{}{}_exec.sh".format(sub_dir, script_basename)
        exec_file_versions = job_helpers.grab_most_recent(exec_file_name_template, return_all=True)

        current_version = []
        current_version.append("#!/usr/bin/env bash\n")
        current_version.append("export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch\n")
        current_version.append("export SITECONFIG_PATH=$VO_CMS_SW_DIR/SITECONF/T2_FR_GRIF_LLR/GRIF-LLR/\n")
        current_version.append("source $VO_CMS_SW_DIR/cmsset_default.sh\n")
        
        condor_args = self._get_condor_args()
        if self.iterOver is not None:
            iter_arg_pos = condor_args.index(self.iterOver.replace("--", ""))
            current_version.append('list=${}\n'.format(iter_arg_pos + 1))

            current_version.append("cleaned_list=$(echo $list | tr -d '[]' | tr ';' '\n')\n")
            current_version.append('while IFS=";" read -r val; do\n')

            batch_strs = [str(batch).replace(", ", ";") for batch in self.batches]

            self.default_args[self.iterOver] = '"$val"'
            

            """write_comm() combines default_args with the command-line arguments,
            verifies that they're accepted by the script, and then returns the python command."""
            python_call = "    " + " ".join(self.true_args.write_comm(self.default_args)) + "\n"

            current_version.append(python_call)
            current_version.append('done <<< "$cleaned_list"')

        else:
            python_call = 'python {}'.format(self.script)

            for arg in sub_args:
                python_call +=' --{} {}'.format(arg, self.default_args[arg])   

            current_version.append(python_call)

        self.sub_file = job_helpers.conditional_write(exec_file_versions, exec_file_name_template, current_version)

    def write_sub_file(self):
        """Writes the .sub script that is submitted to HT Condor.
        Follows the same naming convention and conditional_write()
        procedure as the previous function."""

        log_dir = os.path.join(self.particle_dir, "logs/")

        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        job_file_name_template = "{}/jobs/{}.sub".format(self.particle_dir, script_basename)

        job_file_versions = job_helpers.grab_most_recent(job_file_name_template, return_all=True)

        current_version = []
        current_version.append("executable = {}\n".format(self.sub_file))
        current_version.append("Universe              = vanilla\n")

        arg_keys = "Arguments ="
        condor_args = self._get_condor_args()
        for arg in self._get_condor_args():
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
        
        if len(self.default_args.keys()) > 0:
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

        for d in ("jobs", "logs"):
            d = os.path.join(self.base.particle_dir, d)
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

    job = CondJob("test", config)
    job.prepare_jobs()
    job.launch_jobs()

    '''for particle in ("photons", "electrons", "pions"):
        job = CondJob(particle, config)
        job.prepare_jobs()
        job.launch_jobs()'''