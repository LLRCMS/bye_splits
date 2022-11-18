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
        self.inpath = (Path(__file__).parent.absolute().parent.parent /
                       params.DataFolder / inname )
        self.outpath = (Path(__file__).parent.absolute().parent.parent /
                        params.DataFolder / outname )
        self.dname = 'tc'
        self.var = common.dot_dict({'u': 'waferu', 'v': 'waferv', 'l': 'layer',
                                    'x': 'x', 'y': 'y', 'z': 'z',
                                    'side': 'zside', 'subd': 'subdet'})
        self.newvar = common.dot_dict({'vs': 'waferv_shift', 'c': 'color'})

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
            data = tree.arrays(self.var.values())
            print(data)
            breakpoint()
            sel = (data.zside==1) & (data.subdet==1)
            data = data[sel].drop([self.var.side, self.var.subd], axis=1)
            data = data.loc[~data.layer.isin(params.disconnectedTriggerLayers)]
            data = data.drop_duplicates(subset=[self.var.u, self.var.v, self.var.l])
            data[self.var.v] = data.waferv
            data[self.newvar.vs] = -1 * data.waferv
            data[self.newvar.c] = "#8a2be2"
            
        return data

    def store(self):
        data = self.select()
        with pd.HDFStore(self.outpath, mode='w') as s:
            s[self.dname] = data

    def variables(self):
        res = self.var.copy()
        res.update(self.newvar)
        return res
