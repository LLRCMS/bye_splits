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
        self.iterOver = config["job"]["iterOver"]

    def setup_batches(self):
        total = self.config["job"]["arguments"][self.iterOver]

        vals_per_batch = particle_var(self.particle, "files_per_batch")    

        batches = [total[i: i + vals_per_batch] for i in range(0, len(total), vals_per_batch)]
        
        return batches
    
class CondJobBase:
    def __init__(self, particle, config):
        self.particle = particle
        self.script = config["job"]["script"]
        self.iterOver = config["job"]["iterOver"]
        self.args = config["job"]["arguments"]
        self.queue = config["job"]["queue"]
        self.proxy = config["job"]["proxy"]
        self.local = config["job"]["local"]
        self.user = config["job"]["user"]
        self.particle_dir = particle_var(self.particle, "submit_dir")
        self.batches = JobBatches(particle, config).setup_batches()

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
        
        if len(self.args.keys()) > 0:
            args = ["bash {}".format(self.script)]
            for i, key in enumerate(self.args.keys()):
                args.append("--{} ${}".format(key, i+1))
            args = " ".join(args)
            current_version.append(args)
        else:
            current_version.append("bash {}".format(self.script))
            
        # Write the file only if an identical file doesn't already exist
        self.sub_file = common.conditional_write(submit_file_versions, submit_file_name_template, current_version)

    def prepare_multi_job_condor(self):
        log_dir = "{}logs/".format(self.particle_dir)

        script_basename = os.path.basename(self.script).replace(".sh", "").replace(".py", "")

        job_file_name_template = "{}jobs/{}.sub".format(self.particle_dir, script_basename)

        job_file_versions = common.grab_most_recent(job_file_name_template, return_all=True)

        current_version = []
        current_version.append("executable = {}\n".format(self.sub_file))
        current_version.append("Universe              = vanilla\n")
        if len(self.args) > 0:
            current_version.append("Arguments =")

            for arg in self.args.keys():
                current_version.append(" $({}) ".format(arg))
            current_version.append("\n")

        current_version.append("output = {}{}_C$(Cluster)P$(Process).out\n".format(log_dir, script_basename))
        current_version.append("error = {}{}_C$(Cluster)P$(Process).err\n".format(log_dir, script_basename))
        current_version.append("log = {}{}_C$(Cluster)P$(Process).log\n".format(log_dir, script_basename))
        current_version.append("getenv                = true\n")
        current_version.append("T3Queue = {}\n".format(self.queue))
        current_version.append("WNTag                 = el7\n")
        current_version.append('+SingularityCmd       = ""\n')
        current_version.append("include: /opt/exp_soft/cms/t3/t3queue |\n")

        if len(self.args.keys()) > 0:
            arg_keys = ", ".join(self.args.keys())
            arg_keys = "queue " + arg_keys + " from (\n"
            current_version.append(arg_keys)
            for batch in self.batches:
                sub_args = list(self.args.keys())[1:]
                arg_vals = [self.args[key] for key in sub_args]
                all_vals = ["{}".format(batch).replace(", ", ";")]+arg_vals
                all_vals = ", ".join(all_vals) + "\n"
                current_version.append(all_vals)

            current_version.append(")")

        # Write the file only if an identical file doesn't already exist
        self.submission_file = common.conditional_write(job_file_versions, job_file_name_template, current_version) # Save to launch later

class CondJob:
    def __init__(self, particle, config):
        self.base = CondJobBase(particle=particle, config=config)


    def prepare_jobs(self):

        configs = lambda dir: dir + "configs"
        jobs = lambda dir: dir + "jobs"
        logs = lambda dir: dir + "logs"

        config_dir = configs(self.base.particle_dir)
        job_dir = jobs(self.base.particle_dir)
        log_dir = logs(self.base.particle_dir)

        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.base.prepare_batch_submission()
        self.base.prepare_multi_job_condor()


    def launch_jobs(self):

        if self.base.local == True:
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
        status = subprocess.run(comm)

if __name__ == "__main__":    
    with open(params.CfgPath, "r") as afile:
        config = yaml.safe_load(afile)

    for particle in ("photons", "electrons", "pions"):
        job = CondJob(particle, config)
        job.prepare_jobs()
        job.launch_jobs()