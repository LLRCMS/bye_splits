# coding: utf-8

_all_ = [ 'EventData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import yaml
import numpy as np
import uproot as up
import pandas as pd
import awkward as ak
import functools

from utils import params
from data_handle.base import BaseData

class EventData(BaseData):
    def __init__(self, inname, tag='v0', default_events=[], reprocess=False, logger=None, is_tc=True):
        super().__init__(inname, tag, reprocess, logger, is_tc)

        with open(params.viz_kw['CfgDataPath'], 'r') as afile:
            self.var = yaml.safe_load(afile)['varEvents']

        self.cache = None
        self.events = default_events
        self.cache_events(self.events)
        self.ev_numbers = self._get_event_numbers()

        with open(params.viz_kw['CfgProdPath'], 'r') as afile:
            _cfg = yaml.safe_load(afile)
        self.indata.dir  = _cfg['io']['dir']
        self.indata.tree = _cfg['io']['tree']

    def cache_events(self, events):
        """Read dataset from parquet to cache"""
        if not os.path.exists(self.outpath) or self.reprocess:
            self.store()
        if not isinstance(events, (tuple,list)):
            events = [events]

        ds = ak.from_parquet(self.outpath)
        evmask = False
        for ev in events:
            if not ak.sum(ds.event == ev):
                mes = 'Event {} is not present in file {}.'
                raise RuntimeError(mes.format(ev, self.outpath))
            evmask = evmask | (ds.event==ev)
        ds = ds[evmask]

        dsd = {}
        for k in self.var.keys():
            dsd[k] = ak.to_dataframe(ds[list(self.var[k].values())], how='outer')

        if self.cache is None: #first cache_events() call
            self.cache = dsd
        else:
            for k in self.var.keys():
                self.cache[k] = pd.concat([self.cache[k], dsd[k]], axis=0)
        #self.cache = self.cache.persist() only for dask dataframes

    def _get_event_numbers(self):
        ds = ak.from_parquet(self.outpath)
        return ds.event.tolist()

    def provide(self):
        print('Providing event {} data...'.format(self.tag))
        if not os.path.exists(self.outpath):
            self.store()
        return ak.from_parquet(self.outpath)

    def provide_event(self, event, merge):
        """Provide single event, checking if it is in cache"""
        if event not in self.events:
            self.events += [event]
            self.cache_events(event)

        ret = {}
        for k in self.var.keys():
            tmp = self.cache[k][self.cache[k].event==event]
            tmp = tmp.drop(['event'], axis=1)
            ret[k] = tmp.apply(pd.Series.explode).reset_index(drop=True)
        if merge:
            ret = functools.reduce(
                lambda left,right: pd.concat((left,right), axis=1),
                list(ret.values()))

        return ret

    def provide_random_event(self, merge):
        """Provide a random event"""
        event = np.random.choice(self.ev_numbers)
        return self.provide_event(event, merge), event
        
    def select(self):
        with up.open(self.indata.path, array_cache='550 MB', num_workers=8) as f:
            tree = f[self.indata.tree_path]
            allvars = set([y for x in self.var.values() for y in x.values()])
            data = tree.arrays(filter_name='/' + '|'.join(allvars) + '/',
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
