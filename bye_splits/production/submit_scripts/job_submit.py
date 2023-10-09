#!/usr/bin/env python

import os
import sys
from datetime import datetime

parent_dir = os.path.abspath(__file__ + 5 * "../")
sys.path.insert(0, parent_dir)

from bye_splits.utils import params, common

import subprocess
import yaml

# Read particle specific variables from the YAML file
particle_var = lambda part, var: config["job"][part][var]

class JobBatches:
    def __init__(self, particle, config):
        self.particle = particle
        self.config = config

    def setup_batches(self):

        my_batches = lambda files, files_per_batch: [files[i: i + files_per_batch] for i in range(0, len(files), files_per_batch)]

        read_dir = self.config["job"]["read_dir"]
        files = particle_var(self.particle, "files")
        files_per_batch = particle_var(self.particle, "files_per_batch")
        
        if not read_dir:
            with open(files, "r") as file:
                paths = file.read().splitlines()
        else:
            part_submit_dir = particle_var(self.particle, "submit_dir") + "ntuples/"
            paths = [
                "{}{}".format(part_submit_dir, file) for file in os.listdir(part_submit_dir) if file.startswith("ntuple")
            ]
        
        batches = my_batches(paths, files_per_batch)

        return batches
    
class CondJobBase(JobBatches):
    def __init__(self, particle, config):
        super().__init__(particle, config)
        self.particle_dir = particle_var(self.particle, "submit_dir")
        self.script = config["job"]["script"]
        self.args = config["job"]["arguments"]
        self.queue = config["job"]["queue"]
        self.proxy = config["job"]["proxy"]
        self.local = config["job"]["local"]
        self.user = config["job"]["user"]


    def write_batch_files(self):
        batch_dir = "{}batches/".format(self.particle_dir)
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)
        batch_script_dir = "{}{}/".format(batch_dir, os.path.splitext(os.path.basename(self.script))[0])
        if not os.path.exists(batch_script_dir):
            os.makedirs(batch_script_dir)

        batches = self.setup_batches()
        global current_batch_versions
        current_batch_versions = []
        for i, batch in enumerate(batches):
            out_name = "{}batch_{}.txt".format(batch_script_dir, i)
            written_version = common.grab_most_recent(out_name, return_all=True)
            batch_lines = ["{}\n".format(b) for b in batch]
            current_version = common.conditional_write(written_version, out_name, batch_lines)
            current_batch_versions.append(current_version)


    def prepare_batch_submission(self):
        sub_dir = "{}subs/".format(self.particle_dir)

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        submit_file_name_template = "{}{}_submit.sh".format(sub_dir, script_basename)
        submit_file_versions = common.grab_most_recent(submit_file_name_template, return_all=True)

        current_version = []
        current_version.append("#!/usr/bin/env bash\n")
        current_version.append("workdir={}/bye_splits/production/submit_scripts\n".format(parent_dir))
        current_version.append("cd $workdir\n")
        current_version.append("export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch\n")
        current_version.append("export SITECONFIG_PATH=$VO_CMS_SW_DIR/SITECONF/T2_FR_GRIF_LLR/GRIF-LLR/\n")
        current_version.append("source $VO_CMS_SW_DIR/cmsset_default.sh\n")
        if len(self.args) > 0:
            args = ["bash {}".format(self.script)]
            for i in range(len(self.args)):
                args.append(f"${i+1}")
            args = " ".join(args)
            current_version.append(args)
        else:
            current_version.append("bash {}".format(self.script))
            
        # Write the file only if an identical file doesn't already exist
        global sub_file
        sub_file = common.conditional_write(submit_file_versions, submit_file_name_template, current_version)

    def prepare_multi_job_condor(self):
        log_dir = "{}logs/".format(self.particle_dir)
        
        batch_files = current_batch_versions

        arg_dict = {}
        for arg in self.args:
            if arg=="filename":
                arg_dict[arg] = batch_files
            elif arg=="particles":
                arg_dict[arg] = self.particle
            elif arg=="pileup":
                arg_dict[arg] = "PU0" if "PU0" in batch_files[0] else "PU200"
            else:
                print(f"{arg} is not currently supported.")
                quit()

        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        job_file_name_template = "{}jobs/{}.sub".format(self.particle_dir, script_basename)

        job_file_versions = common.grab_most_recent(job_file_name_template, return_all=True)

        current_version = []
        current_version.append("executable = {}\n".format(sub_file))
        current_version.append("Universe              = vanilla\n")
        if len(self.args) > 0:
            current_version.append("Arguments =")
            for arg in self.args[:-1]:
                current_version.append(" $({}) ".format(arg))
            current_version.append("$({})\n".format(self.args[-1]))
        current_version.append("output = {}{}_C$(Cluster)P$(Process).out\n".format(log_dir, script_basename))
        current_version.append("error = {}{}_C$(Cluster)P$(Process).err\n".format(log_dir, script_basename))
        current_version.append("log = {}{}_C$(Cluster)P$(Process).log\n".format(log_dir, script_basename))
        current_version.append("getenv                = true\n")
        current_version.append("T3Queue = {}\n".format(self.queue))
        current_version.append("WNTag                 = el7\n")
        current_version.append('+SingularityCmd       = ""\n')
        current_version.append("include: /opt/exp_soft/cms/t3/t3queue |\n")
        if len(arg_dict.keys()) > 0:
            arg_keys = [key for key in arg_dict.keys()]
            arg_keys = ", ".join(arg_keys)
            arg_keys = "queue " + arg_keys + " from (\n"
            current_version.append(arg_keys)
            for file in arg_dict["filename"]:
                sub_args = list(arg_dict.keys())[1:]
                arg_vals = [file]+[arg_dict[key] for key in sub_args]
                arg_vals = ", ".join(arg_vals) + "\n"
                current_version.append(arg_vals)
            current_version.append(")")

        # Write the file only if an identical file doesn't already exist
        global submission_file # Save to launch later
        submission_file = common.conditional_write(job_file_versions, job_file_name_template, current_version)

class CondJob(CondJobBase):
    def __init__(self, particle, config):
        super().__init__(particle, config)


    def prepare_jobs(self):
        self.write_batch_files()

        configs = lambda dir: dir + "configs"
        jobs = lambda dir: dir + "jobs"
        logs = lambda dir: dir + "logs"

        config_dir = configs(self.particle_dir)
        job_dir = jobs(self.particle_dir)
        log_dir = logs(self.particle_dir)

        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.prepare_batch_submission()
        self.prepare_multi_job_condor()


    def launch_jobs(self):

        if self.local == True:
            machine = "local"
        else:
            machine = "llrt3.in2p3.fr"

        sub_comm = ["condor_submit"]

        if not self.local:
            print(
                "\nSending {} jobs on {}".format(self.particle, self.queue + "@{}".format(machine))
            )
            print("===============")
            print("\n")

        sub_args = []

        sub_args.append(submission_file)

        if self.local:
            comm = sub_args
        else:
            comm = sub_comm + sub_args

        print(str(datetime.now()), " ".join(comm))
        status = subprocess.run(comm)

if __name__ == "__main__":    
    with open(params.CfgPath, "r") as afile:
        config = yaml.safe_load(afile)

    for particle in ("photons", "electrons", "pions"):
        job = CondJob(particle, config)
        job.prepare_jobs()
        job.launch_jobs()