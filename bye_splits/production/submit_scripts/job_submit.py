#!/usr/bin/env python

import os
import sys
from datetime import datetime

parent_dir = os.path.abspath(__file__ + 5 * "../")
sys.path.insert(0, parent_dir)

from bye_splits.utils import params

import subprocess
import re
import yaml

# Read particle specific variables from the YAML file
particle_var = lambda part, var: config["job"][part][var]

def my_batches(files, files_per_batch):
    return [
        files[i : i + files_per_batch] for i in range(0, len(files), files_per_batch)
    ]

def strip_trail(batches):
    return [[file.rstrip() for file in batch] for batch in batches]

def setup_batches(particle, config):
    read_dir = config["job"]["read_dir"]
    files = particle_var(particle, "files")
    files_per_batch = particle_var(particle, "files_per_batch")
    if not read_dir:
        with open(files, "r") as File:
            lines = File.readlines()
            # readlines() keeps the explicit "/n" character, strip_trail removes this
            batches = strip_trail(my_batches(lines, files_per_batch))
    else:
        part_submit_dir = particle_var(particle, "submit_dir") + "ntuples/"
        paths = [
            "{}{}".format(part_submit_dir, file) for file in os.listdir(part_submit_dir) if file.startswith("ntuple")
        ]
        batches = my_batches(paths, files_per_batch)

    return batches

# Accepts a template for a full path to a file and increments the version
def increment_version(file_path):
    dir, file = os.path.split(file_path)
    base, ext = os.path.splitext(file)
    i = 0
    file_path = "{}/{}_v{}{}".format(dir, base, i, ext)
    while os.path.exists(file_path):
        i += 1
        file_path = "{}/{}_v{}{}".format(dir, base, i, ext)
    return file_path


# Grab the most recent version of the file corresponding to the template file_path
def grab_most_recent(file_path):
    dir, file = os.path.split(file_path)
    base, ext = os.path.splitext(file)
    files = os.listdir(dir)
    version_pattern = re.compile("{}_v(\\d+)\\{}".format(base, ext))
    matches = [version_pattern.search(file) for file in files]
    matches = [match for match in matches if not match is None]
    most_recent = max([int(match.group(1)) for match in matches])
    file_path = dir + "/" + base + "_v" + str(most_recent) + ext
    return file_path


def write_batch_files(batches, particle_dir, script):
    batch_dir = "{}batches/".format(particle_dir)
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    batch_script_dir = "{}{}/".format(batch_dir, os.path.splitext(os.path.basename(script))[0])
    if not os.path.exists(batch_script_dir):
        os.makedirs(batch_script_dir)
    for i, batch in enumerate(batches):
        out_name = "{}batch_{}.txt".format(batch_script_dir, i)
        if not os.path.exists(out_name):
            with open(out_name, "w") as batch_file:
                for path in batch:
                    batch_file.write("{}\n".format(path))


def prepare_batch_submission(particle_dir, script, proxy, user):
    _, script_ext = os.path.splitext(script)

    sub_dir = "{}subs/".format(particle_dir)
    eos_home = "/eos/user/{}/{}/".format(user[0], user)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    script_basename = os.path.basename(script).replace(".sh", "").replace(".py", "")

    submit_file_name_template = "{}{}_submit.sh".format(sub_dir, script_basename)

    submit_file_name = increment_version(submit_file_name_template)
    with open(submit_file_name, "w") as submit_file:
        submit_file.write("#!/usr/bin/env bash\n")
        submit_file.write("workdir={}/bye_splits/production/submit_scripts\n".format(parent_dir))
        submit_file.write("cd $workdir\n")
        submit_file.write("export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch\n")
        submit_file.write("export SITECONFIG_PATH=$VO_CMS_SW_DIR/SITECONF/T2_FR_GRIF_LLR/GRIF-LLR/\n")
        submit_file.write("source $VO_CMS_SW_DIR/cmsset_default.sh\n")
        if script_ext == ".sh":
            submit_file.write("bash {} $1".format(script))
        elif script_ext == ".py":
            submit_file.write("python {} --batch_file $1 --user {}".format(script, user))
    st = os.stat(submit_file_name)
    os.chmod(submit_file_name, st.st_mode | 0o744)


def prepare_multi_job_condor(particle_dir, script, queue, proxy):
    log_dir = "{}logs/".format(particle_dir)
    batch_dir = "{}batches/".format(particle_dir)
    batch_script_dir = "{}{}/".format(batch_dir, os.path.splitext(os.path.basename(script))[0])

    batch_files = os.listdir(batch_script_dir)
    batch_files = [file for file in batch_files if file.endswith(".txt")]

    script_basename = os.path.basename(script).replace(".sh", "").replace(".py", "")

    sub_file = "{}subs/{}_submit.sh".format(particle_dir, script_basename)
    job_file_name_template = "{}jobs/{}.sub".format(particle_dir, script_basename)

    sub_file = grab_most_recent(sub_file)
    job_file_name = increment_version(job_file_name_template)

    if os.path.exists(job_file_name):
        os.remove(job_file_name)
    with open(job_file_name, "w") as job_file:
        job_file.write("executable = {}\n".format(sub_file))
        job_file.write("Universe              = vanilla\n")
        job_file.write("Arguments             = $(filename)\n")
        job_file.write("output = {}{}_C$(Cluster)P$(Process).out\n".format(log_dir, script_basename))
        job_file.write("error = {}{}_C$(Cluster)P$(Process).err\n".format(log_dir, script_basename))
        job_file.write("log = {}{}_C$(Cluster)P$(Process).log\n".format(log_dir, script_basename))
        job_file.write("getenv                = true\n")
        job_file.write("T3Queue = {}\n".format(queue))
        job_file.write("WNTag                 = el7\n")
        job_file.write('+SingularityCmd       = ""\n')
        job_file.write("include: /opt/exp_soft/cms/t3/t3queue |\n")
        job_file.write("queue filename from (\n")
        for file in batch_files:
            job_file.write("{}{}\n".format(batch_script_dir, file))
        job_file.write(")")

    st = os.stat(job_file_name)
    os.chmod(job_file_name, st.st_mode | 0o744)


def prepare_jobs(particle, batches, param):
    # Directories for submission scripts, batches, logs, etc
    particle_dir = particle_var(particle, "submit_dir")

    script = param["job"]["script"]
    queue = param["job"]["queue"]
    proxy = param["job"]["proxy"]
    user = param["job"]["user"]

    write_batch_files(batches, particle_dir, script)

    configs = lambda dir: dir + "configs"
    jobs = lambda dir: dir + "jobs"
    logs = lambda dir: dir + "logs"

    config_dir = configs(particle_dir)
    job_dir = jobs(particle_dir)
    log_dir = logs(particle_dir)

    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    prepare_batch_submission(particle_dir, script, proxy, user)
    prepare_multi_job_condor(particle_dir, script, queue, proxy)


def launch_job(particle, param):
    submit_dir = particle_var(particle, "submit_dir")

    local = param["job"]["local"]
    script = param["job"]["script"]
    queue = param["job"]["queue"]

    if local == True:
        machine = "local"
    else:
        machine = "llrt3.in2p3.fr"

    sub_comm = ["condor_submit"]

    script_basename = os.path.basename(script).replace(".sh", "").replace(".py", "")

    if not local:
        print(
            "\nSending {} jobs on {}".format(particle, queue + "@{}".format(machine))
        )
        print("===============")
        print("\n")

    sub_args = []
    sub_file_name = "{}jobs/{}.sub".format(submit_dir, script_basename)
    sub_file_name = grab_most_recent(sub_file_name)
    sub_args.append(sub_file_name)

    if local:
        comm = sub_args
    else:
        comm = sub_comm + sub_args

    print(str(datetime.now()), " ".join(comm))
    status = subprocess.run(comm)

if __name__ == "__main__":    
    with open(params.CfgPath, "r") as afile:
        config = yaml.safe_load(afile)

    #for particle in ("photons", "electrons", "pions"):
    for particle in ("photons", "electrons"):
        batches = setup_batches(particle, config)
        prepare_jobs(particle, batches, config)
        launch_job(particle, config)
