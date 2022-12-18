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
    def __init__(self, inname='', outname='', tag=None):
        super().__init__(inname, outname, tag)
        
        self.dname = 'tc'
        self.var = {'event': 'event',
                    'wu': 'good_tc_waferu', 'wv': 'good_tc_waferv',
                    'l': 'good_tc_layer',
                    'cv': 'good_tc_cellu', 'cu': 'good_tc_cellv',
                    'en': 'good_tc_energy'}

    def provide(self, reprocess=False):
        assert self.tag is not None
        if not os.path.exists(self.outpath) or reprocess:
            self.store()
        return ak.from_parquet(self.tag + '.parquet')

    def provide_event(self, event, reprocess=False):
        assert self.tag is not None
        if not os.path.exists(self.outpath) or reprocess:
            self.store()
        ds = ak.from_parquet(self.tag + '.parquet')
        fields = ds.fields
        fields.remove('event')
        ds_short = ds[ds.event==event][fields]
        return ak.to_pandas(ds_short).reset_index().drop(['entry', 'subentry'], axis=1)

    def select(self):
        with up.open(self.inpath) as f:
            tree = f[ os.path.join('FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple',
                                   'HGCalTriggerNtuple') ]
            #print(tree.show())
            breakpoint()
            data = tree.arrays(self.var.values())
            
            # data[self.var.v] = data.waferv
            # data[self.newvar.vs] = -1 * data.waferv
            # data[self.newvar.c] = "#8a2be2"
            
        return data

    def store(self):
        data = self.select()
        ak.to_parquet(data, self.tag + '.parquet')
