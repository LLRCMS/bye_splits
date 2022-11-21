# coding: utf-8

_all_ = [ 'EventData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import uproot as up
import pandas as pd
import awkward as ak

from utils import params
from data_handle.base import BaseData

class EventData(BaseData):
    def __init__(self, inname, outname, tag):
        super().__init__(inname, outname, tag)
        
        self.dname = 'tc'
        self.var.update({'tcwu': 'good_tc_waferu', 'tcwv': 'good_tc_waferv',
                         'tcl': 'good_tc_layer',
                         'tcx': 'good_tc_x', 'tcy': 'good_tc_y', 'tcz': 'good_tc_z'})
        self.newvar.update({'vs': 'tcwv_shift', 'c': 'color'})

    def provide(self, reprocess=False):
        if not os.path.exists(self.outpath) or reprocess:
            self.store()
        return ak.from_parquet(self.tag + '.parquet')

    def select(self):
        with up.open(self.inpath) as f:
            tree = f[ os.path.join('FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple',
                                   'HGCalTriggerNtuple') ]
            #print(tree.show())
            data = tree.arrays(self.var.values())
            # data[self.var.v] = data.waferv
            # data[self.newvar.vs] = -1 * data.waferv
            # data[self.newvar.c] = "#8a2be2"
            
        return data

    def store(self):
        data = self.select()
        ak.to_parquet(data, self.tag + '.parquet')

    def variables(self):
        res = self.var.copy()
        res.update(self.newvar)
        return res
