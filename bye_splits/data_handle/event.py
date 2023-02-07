# coding: utf-8

_all_ = [ 'EventData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import yaml
import uproot as up
import numpy as np
import pandas as pd
import awkward as ak

from utils import params
from data_handle.base import BaseData

class EventData(BaseData):
    def __init__(self, inname='', tag='v0', default_events=[], reprocess=False, logger=None):
        super().__init__(inname, tag, reprocess, logger)
        
        with open(params.viz_kw['CfgDataPath'], 'r') as afile:
            cfg = yaml.safe_load(afile)
            self.var = cfg['varEvents']
            
        self.cache = None
        self.events = default_events
        self.cache_events(self.events)
        self.event_numbers = self.get_event_numbers()

    def cache_events(self, events):
        """Read dataset from parquet to cache"""
        if not os.path.exists(self.outpath) or self.reprocess:
            self.store()
        if not isinstance(events, (tuple,list)):
            events = [events]

        ds = ak.from_parquet(self.outpath)
        evmask = False
        for ev in events:
            evmask = evmask | (ds.event==ev)
        ds = ak.to_dataframe(ds[evmask])
        if ds.empty:
            mes = 'Events {} not found (tag = {}).'
            raise RuntimeError(mes.format(' '.join([str(x) for x in events]), self.tag))

        if self.cache is None: #first cache_events() call
            self.cache = ds
        else:
            self.cache = pd.concat([self.cache, ds], axis=0)
        #self.cache = self.cache.persist() only for dask dataframes

    def get_event_numbers(self):
        """Read event numbers from parquet file"""
        ds = ak.from_parquet(self.outpath)
        return ds.event

    def provide(self):
        print('Providing event {} data...'.format(self.tag))
        if not os.path.exists(self.outpath):
            self.store()
        return ak.from_parquet(self.outpath)

    def provide_event(self, event):
        """Provide single event, checking if it is in cache"""
        if event not in self.events:
            self.events += [event]
            self.cache_events(event)
        ret = self.cache[self.cache.event==event].drop(['event'], axis=1)
        ret = ret.apply(pd.Series.explode).reset_index(drop=True)
        return ret
    
    def provide_event_numbers(self):
        return np.random.choice(self.event_numbers)

    def select(self):
        adir = 'FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple'
        atree = 'HGCalTriggerNtuple'

        with up.open(str(self.inpath), array_cache='550 MB', num_workers=8) as f:
            # print(tree.show())
            tree = f[adir + '/' + atree]
            data = tree.arrays(filter_name='/' + '|'.join(self.var.values()) + '/',
                               library='ak',
                               #entry_stop=50, debug
                               )
        # data[self.var.v] = data.waferv
        # data[self.newvar.vs] = -1 * data.waferv
        # data[self.newvar.c] = "#8a2be2"
        return data

    def store(self):
        print('Store event {} data...'.format(self.tag))
        data = self.select()
        if os.path.exists(self.outpath):
            os.remove(self.outpath)
        ak.to_parquet(data, self.outpath)
