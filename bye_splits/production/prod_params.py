import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import utils
from utils import params

from glob import glob

local = False #local machine vs server machine
htcondor = False #whether to submit the script as multiple jobs to HTCondor
if htcondor and local:
    raise ValueError('No submission is possible from your local machine!')

# DeltaR matching threshold
threshold = 0.05

# Select particles that reached the EE section
reachedEE = 2 #0 converted photons; 1: photons that missed HGCAL; 2: photons that hit HGCAL

# Input files
if local:
    files_photons = ['/home/bruno/Downloads/hadd.root']

else:
    if htcondor:
        files_photons = glob.glob('/home/llr/cms/sauvan/DATA_UPG/HGCAL/Ntuples/study_autoencoder/3_22_1/SinglePhoton_PT2to200/GammaGun_Pt2_200_PU0_HLTWinter20_std_ae_xyseed/210430_091126/ntuple*.root')
    else:
        files_photons = [ str(Path('/data_CMS') / 'cms' / os.environ['USER'] / 'TriggerCells' / 'hadd.root') ]

# Pick one of the different algos trees to retrieve the gen information
gen_tree = 'FloatingpointThresholdDummyHistomaxnoareath20Genclustersntuple/HGCalTriggerNtuple'

# Store only information on the best match; it removes duplicated clusters
bestmatch_only = False

if htcondor:
    out_dir = '/home/llr/cms/sauvan/DATA_UPG/HGCAL/Dataframes/study_autoencoder/3_22_1/electron_photon_signaldriven/'
    file_per_batch_electrons = 5
    file_per_batch_pions = 2
    file_per_batch_photons = 2
else:
    out_dir = params.base_kw['BasePath']
        
out_name = 'gen_cl3d_tc.hdf5'

# List of ECON algorithms
fes = [ 'ThresholdDummyHistomaxnoareath20',
        #'ThresholdTruncation120default'Histomax''
        #'ThresholdTruncation120flat'Histomaxxydr015
       ]

ntuple_template = 'Floatingpoint{fe}Genclustersntuple/HGCalTriggerNtuple'
algo_trees = {}
for fe in fes:
    algo_trees[fe] = ntuple_template.format(fe=fe)
    assert(algo_trees[fe] == gen_tree) #remove ass soon as other algorithms are considered