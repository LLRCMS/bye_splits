# coding: utf-8

_all_ = [ 'EventData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import uproot as up
import pandas as pd

from utils import params, common

class EventData:
    def __init__(self, inname, outname):
        self.inpath = Path('/eos/user/b/bfontana/FPGAs/new_algos/') / inname
        self.outpath = Path('/eos/user/b/bfontana/FPGAs/new_algos/') / outname
        
        self.dname = 'tc'
        self.var = common.dot_dict({'tcwu': 'good_tc_waferu', 'tcwv': 'good_tc_waferv',
                                    'tcl': 'good_tc_layer',
                                    'tcx': 'good_tc_x', 'tcy': 'good_tc_y', 'tcz': 'good_tc_z'})
        self.newvar = common.dot_dict({'vs': 'tcwv_shift', 'c': 'color'})

    def provide(self, reprocess=False):
        if not os.path.exists(self.outpath) or reprocess:
            self.store()
        with pd.HDFStore(self.outpath, mode='r') as s:
            res = s[self.dname]
        return res

    def select(self):
        with up.open(self.inpath) as f:
            tree = f[ os.path.join('FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple',
                                   'HGCalTriggerNtuple') ]
            #print(tree.show())
            data = tree.arrays(self.var.values(), library='np')
            # data[self.var.v] = data.waferv
            # data[self.newvar.vs] = -1 * data.waferv
            # data[self.newvar.c] = "#8a2be2"
            
        return data

    def store(self):
        data = self.select()
        with pd.HDFStore(self.outpath, mode='w') as s:
            breakpoint()
            df = pd.DataFrame({k:v[0] for k,v in data.items()})
            s[self.dname] = df

    def variables(self):
        res = self.var.copy()
        res.update(self.newvar)
        return res
