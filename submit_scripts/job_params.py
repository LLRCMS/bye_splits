# Permissions
user = "iehle"
proxy = "~/.t3/proxy.cert"
queue = "short"

# Use /eos storage vs. local
store_eos = True

# Run locally instead of submitting the job to HTCondor
local = False

# Directories
base_dir = "/data_CMS/cms/ehle/L1HGCAL/PU200/"
submit_dir = "/grid_mnt/vol_home/llr/cms/ehle/NewRepos/bye_splits/submit_scripts/"

# Scripts
script = f"{submit_dir}skim_pu_multicluster.sh"

# Submission and output directories
phot_submit_dir = f"{base_dir}photons/"
el_submit_dir = f"{base_dir}electrons/"
pion_submit_dir = f"{base_dir}pions/"

read_dir = True
if read_dir:
    files_photons = f"{phot_submit_dir}ntuples/"
    files_electrons = f"{el_submit_dir}ntuples/"
else:
    # File paths stored on .txt files
    files_photons = f"{phot_submit_dir}photon_ntuples.txt"
    files_electrons = f"{el_submit_dir}electron_ntuples.txt"
files_pions = []


# Number of files to process in each batch
files_per_batch_elec = 10
files_per_batch_phot = 10
files_per_batch_pion = 10
