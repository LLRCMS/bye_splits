# coding: utf-8

_all_ = [ 'data_handle' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

from data_handle.geometry import GeometryData
from data_handle.event import EventData

def handle(mode, particle=None):
    modes = ('geom', 'event')
    if mode != 'geom' and particle is None:
        raise ValueError('Please provide the particle type.')

    datasets = {'photons'  : {'in': 'skim_photon_0PU_bc_stc_hadd.root',
                              'out': 'out_photon_0PU_bc_stc_hadd.hdf5'},
                'electrons': {'in': 'skim_electron_0PU_bc_stc_hadd.root',
                              'out': 'out_electron_0PU_bc_stc_hadd.hdf5'},
                }
    if mode == modes[0]:
        obj = GeometryData(inname='test_triggergeom.root', outname='geom.hdf5')
    elif mode == modes[1]:
        obj = EventData(inname=datasets['n'][particle], outname=datasets['out'][particle])
    else:
        raise ValueError('Mode {} not supported. Pick one of the following: {}'.format(mode, modes))
    return obj
