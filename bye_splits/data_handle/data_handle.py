# coding: utf-8

_all_ = [ 'data_handle' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

from data_handle.geometry import GeometryData
from data_handle.event import EventData

def handle(mode):
    modes = ('geom', 'event')
    if mode == modes[0]:
        obj = GeometryData(inname='test_triggergeom.root', outname='geom.hdf5')
    elif mode == modes[1]:
        obj = EventData(inname='photon_0PU_bc_stc_hadd.root', outname='geom.hdf5')
    else:
        raise ValueError('Mode {} not supported. Pick one of the following: {}'.format(mode, modes))
    return obj
