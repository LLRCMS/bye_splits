# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
from glob import glob
import functools
import operator
import re

NbinsRz = 42
NbinsPhi = 216
MinROverZ = 0.076
MaxROverZ = 0.58
MinPhi = -np.pi
MaxPhi = +np.pi
DataFolder = 'data'

base_kw = {
    'NbinsRz': NbinsRz,
    'NbinsPhi': NbinsPhi,
    'MinROverZ': MinROverZ,
    'MaxROverZ': MaxROverZ,
    'MinPhi': MinPhi,
    'MaxPhi': MaxPhi,

    'LayerEdges': [0,42],
    'IsHCAL': False,

    'DataFolder': DataFolder,
    'FesAlgos': ['ThresholdDummyHistomaxnoareath20'],
    'BasePath': Path(__file__).parents[2] / DataFolder,

    'OutPath': Path(__file__).parents[2] / 'out',

    'RzBinEdges': np.linspace( MinROverZ, MaxROverZ, num=NbinsRz+1 ),
    'PhiBinEdges': np.linspace( MinPhi, MaxPhi, num=NbinsPhi+1 ),
}

def set_dictionary(adict):
    adict.update(base_kw)
    return adict

if len(base_kw['FesAlgos'])!=1:
    raise ValueError('The event number in the cluster task'
                     ' assumes there is only on algo.\n'
                     'The script must be adapted.')

threshold = 0.05
delta_r_coefs = (0.0,threshold,50)

ntuple_templates = {'photon': 'Floatingpoint{fe}Genclustersntuple/HGCalTriggerNtuple','pion':'hgcalTriggerNtuplizer/HGCalTriggerNtuple'}
#ntuple_templates = {'pion':'hgcalTriggerNtuplizer/HGCalTriggerNtuple'}
algo_trees = {}
for fe in base_kw['FesAlgos']:
    inner_trees = {}
    for key, val in ntuple_templates.items():
        inner_trees[key] = val.format(fe=fe)
    algo_trees[fe] = inner_trees
    #assert(algo_trees[fe] == gen_tree) #remove ass soon as other algorithms are considered

coefs = [(coef,0)*52 for coef in np.linspace(delta_r_coefs[0], delta_r_coefs[1], delta_r_coefs[2])]
coef_dict = {}
for i,coef in enumerate(coefs):
    coef_key = 'coef_'+str(i)
    coef_dict[coef_key] = coef

def transform(nested_list):
    regular_list=[]
    for ele in nested_list:
        if type(ele) is list:
            regular_list.append(ele)
        else:
            regular_list.append([ele])
    return regular_list


def create_out_names(files,trees):
    output_file_names = {}
    for key in files.keys():
        if isinstance(files[key], str):
            files[key] = [files[key]]
        tree = trees[key]
        #output_file_names[key] = ['{}_gen_cl3d_tc_{}'.format(key,re.split('.root|/',file)[-2]) for file in files[key]]
        output_file_names[key] = ['gen_cl3d_tc_{}_{}'.format(base_kw['FesAlgos'][0],re.split('.root|/',file)[-2]) for file in files[key]]
    return output_file_names

files = {'photon': '/data_CMS/cms/alves/TriggerCells/photon_0PU_truncation_hadd.root', 'pion': glob('/data_CMS_upgrade/sauvan/HGCAL/2210_Ehle_clustering-studies/SinglePion_PT0to200/PionGun_Pt0_200_PU0_HLTSummer20ReRECOMiniAOD_2210_clustering-study_v3-29-1/221018_121053/ntuple*.root')}
#files = {'pion': glob('/data_CMS_upgrade/sauvan/HGCAL/2210_Ehle_clustering-studies/SinglePion_PT0to200/PionGun_Pt0_200_PU0_HLTSummer20ReRECOMiniAOD_2210_clustering-study_v3-29-1/221018_121053/ntuple*.root')}
gen_trees = {'photon': 'FloatingpointThresholdDummyHistomaxnoareath20Genclustersntuple/HGCalTriggerNtuple', 'pion':'hgcalTriggerNtuplizer/HGCalTriggerNtuple'}
#gen_trees = {'pion':'hgcalTriggerNtuplizer/HGCalTriggerNtuple'}

match_kw = set_dictionary(
    { 'Files': files,
      'GenTrees': gen_trees,
      'AlgoTrees': algo_trees,
      'File': None, # The following four values are chosen from their respective dicts in the matching process
      'GenTree': None,
      'AlgoTree': None,
      'OutFile': None,
      'BestMatch': False,
      'ReachedEE': 2, #0 converted photons; 1: photons that missed HGCAL; 2: photons that hit HGCAL
      'CoeffAlgos': coef_dict,
      'Threshold': threshold}
)

# fill task
fill_kw = set_dictionary(
    {'FillInFiles' : create_out_names(files, match_kw['GenTrees']),
     'FillIn'      : None, # To be chosen during the fill process
     'FillOut'     : 'fill',
     'FillOutComp' : 'fill_comp',
     'FillOutPlot' : 'fill_plot' }
     )

# optimization task
opt_kw = set_dictionary(
    { 'Epochs': 99999,
      'KernelSize': 10,
      'WindowSize': 3,
      'InFile': None,
      'OptIn': 'triggergeom_condensed', #Needs to be adjusted because this will change for each starting file
      'OptEnResOut': 'opt_enres',
      'OptPosResOut': 'opt_posres',
      'OptCSVOut': 'stats',
      'FillOutPlot': fill_kw['FillOutPlot'],
      'Pretrained': False,
      'LayersToOptimize': [x for x in range(9)],
    }
)

# smooth task
smooth_kw = set_dictionary(
    { #copied from L1Trigger/L1THGCal/python/hgcalBackEndLayer2Producer_cfi.py
        'BinSums': (13,               # 0
                    11, 11, 11,       # 1 - 3
                    9, 9, 9,          # 4 - 6
                    7, 7, 7, 7, 7, 7,  # 7 - 12
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 13 - 27
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3  # 28 - 41
                    ),
        'SeedsNormByArea': False,
        'AreaPerTriggerCell': 4.91E-05,
        'SmoothIn': fill_kw['FillOut'],
        'SmoothOut': 'smooth' }
    )

# seed task
seed_kw = set_dictionary(
    { 'SeedIn': smooth_kw['SmoothOut'],
      'SeedOut': 'seed',
      'histoThreshold': 20.,
      'WindowPhiDim': 1}
    )

# cluster task
cluster_kw = set_dictionary(
    { 'ClusterInTC': fill_kw['FillOut'],
      'ClusterInSeeds': seed_kw['SeedOut'],
      'ClusterOutPlot': 'cluster_plot',
      'ClusterOutValidation': 'cluster_validation',
      'CoeffA': ( (0.015,)*7 + (0.020,)*7 + (0.030,)*7 + (0.040,)*7 + #EM
                  (0.040,)*6 + (0.050,)*6 + # FH
                  (0.050,)*12 ), # BH
      'CoeffB': 0,
      'MidRadius': 2.3,
      'PtC3dThreshold': 0.5,
      'ForEnergy': False,
      'EnergyOut': 'cluster_energy',
      'GenPart': fill_kw['FillIn']}
)

# validation task
validation_kw = set_dictionary(
    { 'ClusterOutValidation': cluster_kw['ClusterOutValidation'],
      'FillOutComp' : fill_kw['FillOutComp'],
      'FillOut': fill_kw['FillOut'] }
)

# energy task
energy_kw = set_dictionary(
    { 'ClusterIn': cluster_kw['ClusterOutValidation'],
      'Coeff': cluster_kw['CoeffA'],
      'ReInit': False, # If true, ../scripts/en_per_deltaR.py will create an .hdf5 file containing energy info.
      'Coeffs': delta_r_coefs, #tuple containing (coeff_start, coeff_end, num_coeffs)
      'EnergyIn': cluster_kw['EnergyOut'],
      'EnergyOut': 'energy_out',
      'EnergyPlot': 'energy_plot',
      'BestMatch': True,
      'MatchFile': False,
      'MakePlot': True}
)
