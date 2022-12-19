# coding: utf-8

_all_ = [ 'EventData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import yaml
import uproot as up
import pandas as pd
import awkward as ak
import dask.dataframe as dd

from utils import params
from data_handle.base import BaseData

class EventData(BaseData):
    def __init__(self, inname='', tag='v0', default_events=[]):
        super().__init__(inname, tag)
        
        with open(params.viz_kw['CfgEventPath'], 'r') as afile:
            cfg = yaml.safe_load(afile)
            self.var = cfg['varEvents']
            
        self.cache = None
        self.events = default_events
        self.cache_events(self.events)

    def cache_events(self, events):
        """Read dataset from parquet to cache"""
        assert self.tag is not None
        if not os.path.exists(self.outpath):
            self.store()
        if not isinstance(events, (tuple,list)):
            events = list(events)

        ds = dd.read_parquet(self.outpath, engine='pyarrow')
        ds = ds[ds.event.isin(events)]

        if not self.cache: #first cache_events() call
            self.cache = ds
        else:
            self.cache = dd.concat([self.cache, ds], axis=0)
        self.cache = self.cache.persist()

    def provide(self, reprocess=False):
        assert self.tag is not None
        if not os.path.exists(self.outpath) or reprocess:
            self.store()
        return ak.from_parquet(self.outpath)

    def provide_event(self, event, reprocess=False):
        """Provide single event, checking if it is in cache"""
        if event not in self.events:
            self.events += [event]
            self.cache_events(event)
        ret = self.cache[self.cache.event==event].compute()
        ret = ret.apply(pd.Series.explode)
        return ret
    
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
        print('Storing...')
        data = self.select()
        ak.to_parquet(data, self.tag + '.parquet')
