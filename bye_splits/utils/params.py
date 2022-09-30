# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np

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

    #os.path.join(os.environ['PWD'], DataFolder),
    'OutPath': os.path.join(os.environ['PWD'], 'out'),

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

# fill task
fill_kw = set_dictionary(
    {'FillIn'      : 'gen_cl3d_tc',
     'FillOut'     : 'fill',
     'FillOutComp' : 'fill_comp',
     'FillOutPlot' : 'fill_plot' }
     )

# optimization task
opt_kw = set_dictionary(
    { 'Epochs': 99999,
      'KernelSize': 10,
      'WindowSize': 3,
      'OptIn': 'triggergeom_condensed',
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
	  'ClusterOutPlot': 'cluster_validation',
      'ClusterOutValidation': 'cluster_plot',
      'CoeffA': ( (0.015,)*7 + (0.020,)*7 + (0.030,)*7 + (0.040,)*7 + #EM
                  (0.040,)*6 + (0.050,)*6 + # FH
                  (0.050,)*12 ), # BH
      'CoeffB': 0,
      'MidRadius': 2.3,
      'PtC3dThreshold': 0.5 }
)

# validation task
validation_kw = set_dictionary(
    { 'ClusterOutValidation': cluster_kw['ClusterOutValidation'],
      'FillOutComp' : fill_kw['FillOutComp'],
      'FillOut': fill_kw['FillOut'] }
)

