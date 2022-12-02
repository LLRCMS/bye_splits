from glob import glob
import itertools

local = True #local machine vs server machine
htcondor = False #whether to submit the script as multiple jobs to HTCondor
if htcondor and local:
    raise ValueError('No submission is possible from your local machine!')

# DeltaR matching threshold
#THIS HAS TO BE CHANGED FOR MY CASE
threshold = 0.05

# Select particles that reached the EE section
reachedEE = 2 #0 converted photons; 1: photons that missed HGCAL; 2: photons that hit HGCAL

# Input files
if local:
    files_photons = ['/data_CMS/cms/alves/TriggerCells/hadd.root']
    files_electrons = []
    files_pions = []

else:
    files_photons = glob('/home/llr/cms/sauvan/DATA_UPG/HGCAL/Ntuples/study_autoencoder/3_22_1/SinglePhoton_PT2to200/GammaGun_Pt2_200_PU0_HLTWinter20_std_ae_xyseed/210430_091126/ntuple*.root')
    files_electrons = []
    files_pions = []
    if not htcondor:
        files_photons = files_photons[:1] if len(files_photons)>0 else []
        files_electrons = files_electrons[:1] if len(files_electrons)>0 else []
        files_pions = files_pions[:1] if len(files_pions)>0 else []

# Pick one of the different algos trees to retrieve the gen information
gen_tree = 'FloatingpointThresholdDummyHistomaxnoareath20Genclustersntuple/HGCalTriggerNtuple'

# Store only information on the best match
bestmatch_only = False #best match removes duplicated clusters

if local:
    output_dir = '.'
else:
    output_dir = '/home/llr/cms/sauvan/DATA_UPG/HGCAL/Dataframes/study_autoencoder/3_22_1/electron_photon_signaldriven/'
    file_per_batch_electrons = 5
    file_per_batch_pions = 2
    file_per_batch_photons = 2
output_file_name = 'gen_cl3d_tc.hdf5'

# List of ECON algorithms
fes = [ 'ThresholdDummyHistomaxnoareath20',
        #
       ]
# other possibilities:
# 'ThresholdTruncation120default'Histomax''
# 'ThresholdTruncation120flat'Histomaxxydr015

ntuple_template = 'Floatingpoint{fe}Genclustersntuple/HGCalTriggerNtuple'
algo_trees = {}
for fe in fes:
    algo_trees[fe] = ntuple_template.format(fe=fe)
    assert(algo_trees[fe] == gen_tree) #remove ass soon as other algorithms are considered
