#!/usr/bin/env python

import os
import sys
from datetime import datetime

parent_dir = os.path.abspath(__file__ + 3 * "../")
sys.path.insert(0, parent_dir)

import subprocess
import optparse
import re


def my_batches(files, files_per_batch):
    return [
        files[i : i + files_per_batch] for i in range(0, len(files), files_per_batch)
    ]


def strip_trail(batches):
    return [[file.rstrip() for file in batch] for batch in batches]


def setup_batches(files, files_per_batch=10, read_dir=False):
    if not read_dir:
        with open(files, "r") as File:
            Lines = File.readlines()
            # readlines() keeps the explicit "/n" character, strip_trail removes this
            batches = strip_trail(my_batches(Lines, files_per_batch))
    else:
        paths = [
            f"{files}{file}" for file in os.listdir(files) if file.startswith("ntuple")
        ]
        batches = my_batches(paths, files_per_batch)

    return batches


# Accepts a template for a full path to a file and increments the version
def increment_version(file_path):
    dir, file = os.path.split(file_path)
    base, ext = os.path.splitext(file)
    i = 0
    file_path = f"{dir}/{base}_v{i}{ext}"
    while os.path.exists(file_path):
        i += 1
        file_path = f"{dir}/{base}_v{i}{ext}"
    return file_path


# Grab the most recent version of the file corresponding to the template file_path
def grab_most_recent(file_path):
    dir, file = os.path.split(file_path)
    base, ext = os.path.splitext(file)
    files = os.listdir(dir)
    version_pattern = re.compile(f"{base}_v(\\d+)\\{ext}")
    matches = [version_pattern.search(file) for file in files]
    matches = [match for match in matches if not match is None]
    most_recent = max([int(match.group(1)) for match in matches])
    file_path = dir + "/" + base + "_v" + str(most_recent) + ext
    return file_path


def write_batch_files(batches, particle_dir, script):
    batch_dir = f"{particle_dir}batches/"
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    batch_script_dir = f"{batch_dir}{os.path.splitext(os.path.basename(script))[0]}/"
    if not os.path.exists(batch_script_dir):
        os.makedirs(batch_script_dir)
    for i, batch in enumerate(batches):
        out_name = f"{batch_script_dir}batch_{i}.txt"
        if not os.path.exists(out_name):
            with open(out_name, "w") as batch_file:
                for path in batch:
                    batch_file.write(f"{path}\n")


def prepare_batch_submission(particle_dir, script, proxy, user):
    _, script_ext = os.path.splitext(script)

    sub_dir = f"{particle_dir}subs/"
    eos_home = f"/eos/user/{user[0]}/{user}/"
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    script_basename = os.path.basename(script).replace(".sh", "").replace(".py", "")

    submit_file_name_template = f"{sub_dir}{script_basename}_submit.sh"

    submit_file_name = increment_version(submit_file_name_template)
    with open(submit_file_name, "w") as submit_file:
        print("#!/usr/bin/env bash", file=submit_file)
        print(
            f"workdir={parent_dir}/submit_scripts",
            file=submit_file,
        )
        print("cd $workdir", file=submit_file)
        print("export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch", file=submit_file)
        print(
            "export SITECONFIG_PATH=$VO_CMS_SW_DIR/SITECONF/T2_FR_GRIF_LLR/GRIF-LLR/",
            file=submit_file,
        )
        print("source $VO_CMS_SW_DIR/cmsset_default.sh", file=submit_file)
        """print('export EOS_MGM_URL="root://eosuser.cern.ch/"', file=submit_file)
        print(f"export EOS_HOME={eos_home}", file=submit_file)
        print("export X509_USER_PROXY={}".format(proxy), file=submit_file)
        print(
            "export KRB5CCNAME=FILE:/home/llr/cms/ehle/.krb5c/ehle",
            file=submit_file,
        )
        print(f"/opt/exp_soft/cms/t3/eos-login -username {user} -wn", file=submit_file)"""
        if script_ext == ".sh":
            print(f"bash {script} ${1}", file=submit_file)
        elif script_ext == ".py":
            print(f"python {script} --batch_file ${1} --user {user}", file=submit_file)
    st = os.stat(submit_file_name)
    os.chmod(submit_file_name, st.st_mode | 0o744)


def prepare_multi_job_condor(particle_dir, script, queue, proxy):
    log_dir = f"{particle_dir}logs/"
    batch_dir = f"{particle_dir}batches/"
    batch_script_dir = f"{batch_dir}{os.path.splitext(os.path.basename(script))[0]}/"

    batch_files = os.listdir(batch_script_dir)
    batch_files = [file for file in batch_files if file.endswith(".txt")]

    script_basename = os.path.basename(script).replace(".sh", "").replace(".py", "")

    sub_file = f"{particle_dir}subs/{script_basename}_submit.sh"
    job_file_name_template = f"{particle_dir}jobs/{script_basename}.sub"

    sub_file = grab_most_recent(sub_file)
    job_file_name = increment_version(job_file_name_template)

    if os.path.exists(job_file_name):
        os.remove(job_file_name)
    with open(job_file_name, "w") as job_file:
        print(f"executable           = {sub_file}", file=job_file)
        print("Universe              = vanilla", file=job_file)
        print("Arguments             = $(filename)", file=job_file)
        print(
            f"output               = {log_dir}{script_basename}_C$(Cluster)P$(Process).out",
            file=job_file,
        )
        print(
            f"error                = {log_dir}{script_basename}_C$(Cluster)P$(Process).err",
            file=job_file,
        )
        print(
            f"log                  = {log_dir}{script_basename}_C$(Cluster)P$(Process).log",
            file=job_file,
        )
        print("getenv                = true", file=job_file)
        print(f"T3Queue              = {queue}", file=job_file)
        print("WNTag                 = el7", file=job_file)
        print('+SingularityCmd       = ""', file=job_file)
        print("include: /opt/exp_soft/cms/t3/t3queue |", file=job_file)
        print("queue filename from (", file=job_file)
        for file in batch_files:
            print(f"{batch_script_dir}{file}", file=job_file)
        print(")", file=job_file)

    st = os.stat(job_file_name)
    os.chmod(job_file_name, st.st_mode | 0o744)


def prepare_jobs(param, batches_phot, batches_elec, batches_pion):
    files_pions = param.files_pions

    # Directories for submission scripts, batches, logs, etc
    phot_dir = param.phot_submit_dir
    elec_dir = param.el_submit_dir
    pion_dir = param.pion_submit_dir

    script = param.script
    proxy = param.proxy
    queue = param.queue
    user = param.user

    write_batch_files(batches_phot, phot_dir, script)
    write_batch_files(batches_elec, elec_dir, script)
    write_batch_files(batches_pion, pion_dir, script)

    configs = lambda dir: dir + "configs"
    jobs = lambda dir: dir + "jobs"
    logs = lambda dir: dir + "logs"

    for dir in [phot_dir, elec_dir, pion_dir]:
        config_dir = configs(dir)
        job_dir = jobs(dir)
        log_dir = logs(dir)
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    prepare_batch_submission(phot_dir, script, proxy, user)
    prepare_multi_job_condor(phot_dir, script, queue, proxy)

    prepare_batch_submission(elec_dir, script, proxy, user)
    prepare_multi_job_condor(elec_dir, script, queue, proxy)

    if len(files_pions) > 0:
        prepare_batch_submission(pion_dir, script, proxy, user)
        prepare_multi_job_condor(pion_dir, script, queue, proxy)

    return phot_dir, elec_dir, pion_dir


def launch_job(particle, submit_dir, script, user, queue, proxy, local=True):
    if local == True:
        machine = "local"
    else:
        machine = "llrt3.in2p3.fr"

    sub_comm = ["condor_submit"]

    script_basename = os.path.basename(script).replace(".sh", "").replace(".py", "")

    if not local:
        print(
            "\nSending {0} jobs on {1}".format(particle, queue + "@{}".format(machine))
        )
        print("===============")
        print("\n")

    sub_args = []
    sub_file_name = f"{submit_dir}jobs/{script_basename}.sub"
    sub_file_name = grab_most_recent(sub_file_name)
    sub_args.append(sub_file_name)

    if local:
        comm = sub_args
    else:
        comm = sub_comm + sub_args

    print(str(datetime.now()), " ".join(comm))
    status = subprocess.run(comm)


def main(parameters_file):
    import importlib
    import sys

    sys.path.append(parent_dir)
    parameters_file = parameters_file.replace("/", ".").replace(".py", "")

    parameters = importlib.import_module(parameters_file)

    working_dir = parameters.working_dir

    local = parameters.local
    user = parameters.user
    proxy = parameters.proxy
    queue = parameters.queue

    read_dir = parameters.read_dir

    files_electrons = parameters.files_electrons
    files_photons = parameters.files_photons
    files_pions = parameters.files_pions
    files_per_batch_elec = parameters.files_per_batch_elec
    files_per_batch_phot = parameters.files_per_batch_phot
    files_per_batch_pion = parameters.files_per_batch_pion

    script = parameters.script

    batches_elec = setup_batches(
        files_electrons, files_per_batch_elec, read_dir=read_dir
    )
    batches_phot = setup_batches(files_photons, files_per_batch_phot, read_dir=read_dir)
    batches_pion = []

    if len(files_pions) > 0:
        batches_pion = setup_batches(files_pions, files_per_batch_pion)

    phot_dir, elec_dir, pion_dir = prepare_jobs(
        parameters, batches_phot, batches_elec, batches_pion
    )

    launch_job(
        "photons", phot_dir, script, user=user, queue=queue, proxy=proxy, local=local
    )
    launch_job(
        "electrons", elec_dir, script, user=user, queue=queue, proxy=proxy, local=local
    )
    if len(files_pions) > 0:
        launch_job(
            "pions", pion_dir, script, user=user, queue=queue, proxy=proxy, local=local
        )


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option(
        "--cfg", type="string", dest="param_file", help="select the parameter file"
    )
    (opt, args) = parser.parse_args()
    parameters = opt.param_file
    main(parameters)
