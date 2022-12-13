import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
import utils
from utils import params

from glob import glob
import pickle

local = True #local machine vs server machine
htcondor = False #whether to submit the script as multiple jobs to HTCondor
particle = 'photon'
algo = 'best_choice'
pu = False
assert particle in ('photon', 'pion', 'electron')
assert algo in ('best_choice', 'truncation')

if htcondor and local:
    raise ValueError('No submission is possible from your local machine!')

# DeltaR matching threshold
threshold = 0.05

# Select particles that reached the EE section
reachedEE = 2 #0 converted photons; 1: photons that missed HGCAL; 2: photons that hit HGCAL

if local:
    base = '/data_CMS/cms/ehle/L1HGCAL/'
    file_ext = '_200PU_bc_stc_hadd'
    files = {'photon'   : base+'photon'+file_ext,
             'electron' : base+'electron'+file_ext}
    files = files[particle]

else:
    if htcondor:
        files_photons = glob.glob('/home/llr/cms/sauvan/DATA_UPG/HGCAL/Ntuples/study_autoencoder/3_22_1/SinglePhoton_PT2to200/GammaGun_Pt2_200_PU0_HLTWinter20_std_ae_xyseed/210430_091126/ntuple*.root')
    else:
        base = Path('/grid_mnt/vol_home/llr/cms/') / 'alves' / 'CMSSW_12_5_0_pre1' / 'src/bye_splits/data/new_algos/'
        files = {'photon'   : [str(base / 'skim_photon_0PU_bc_stc_hadd.root')],
                 'electron' : [str(base / 'electron_0PU_bc_stc_hadd.root')],
                 'pion'     : [str(base / 'pion_0PU_bc_stc_hadd.root')]}
        files = files[particle]

# Pick one of the different algos trees to retrieve the gen information
gen_tree = {'best_choice': 'FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple',
            'truncation': 'FloatingpointThresholdDummyHistomaxnoareath20Genclustersntuple'}
gen_tree = os.path.join(gen_tree[algo], 'HGCalTriggerNtuple')

# Store only information on the best match; it removes duplicated clusters
bestmatch_only = False

if htcondor:
    out_dir = '/home/llr/cms/sauvan/DATA_UPG/HGCAL/Dataframes/study_autoencoder/3_22_1/electron_photon_signaldriven/'
    file_per_batch_electrons = 5
    file_per_batch_pions = 2
    file_per_batch_photons = 2
else:
    out_dir = params.base_kw['BasePath']

out_name = 'summ_{}_{}.hdf5'.format(particle, algo)
algo_trees = [gen_tree]
