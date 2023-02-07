# coding: utf-8

_all_ = [ 'data_handle' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import yaml

from utils import params
from data_handle.geometry import GeometryData
from data_handle.event import EventData

class EventDataParticle:
    def __init__(self, particles, tag, reprocess=False, debug=False, logger=None):
        assert particles in ('photons', 'electrons', 'pions')
        self.particles = particles
        self.tag = self.particles + '_' + tag
        with open(params.viz_kw['CfgDataPath'], 'r') as afile:
            self.config = yaml.safe_load(afile)

        data_suffix = 'skim' + ('_small' if debug else '')
        in_name = '_'.join((data_suffix, self.particles, '0PU_bc_stc_hadd.root'))
        default_events = self.config['defaultEvents'][self.particles]
        self.data = EventData(in_name, self.tag + '_debug' * debug,
                              default_events, reprocess=reprocess, logger=logger)

    def provide_event_numbers(self):
        return self.data.provide_event_numbers()
   
    def provide_event(self, event):
        return self.data.provide_event(event)
