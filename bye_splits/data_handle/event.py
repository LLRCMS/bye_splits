# coding: utf-8

_all_ = ["EventData"]

import os
from pathlib import Path
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
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
    def __init__(self, inname, tag="v0", default_events=[], reprocess=False,
                 logger=None, is_tc=True, set_default_events=False,
                 cfgkey='prod'):
        super().__init__(inname, tag, reprocess, logger, is_tc, cfgkey)

        with open(params.CfgPaths["data"], "r") as afile:
            self.var = yaml.safe_load(afile)["varEvents"]

        self.cache = None
        self.events = []
        if set_default_events:
            self.events = default_events
            self.cache_events(self.events)

        self.ev_numbers = self._get_event_numbers()
        self.rng = np.random.default_rng()

    def _add_events(self, events):
        events = self._convert_to_list(events)
        self.events += events
        self.cache_events(events)

    def cache_events(self, events):
        """Read dataset from parquet to cache"""
        if not os.path.exists(self.outpath) or self.reprocess:
            self.store()
        events = self._convert_to_list(events)

        ds = ak.from_parquet(self.outpath)
        ds = self._event_mask(ds, events)

        dsd = {}
        for k in self.var.keys():
            dsd[k] = ak.to_dataframe(ds[list(self.var[k].values())], how="outer")

        if self.cache is None:  # first cache_events() call
            self.cache = dsd
        else:
            for k in self.var.keys():
                self.cache[k] = pd.concat([self.cache[k], dsd[k]], axis=0)

        # self.cache = self.cache.persist() only for dask dataframes

    def _convert_to_list(self, events):
        """Converts a variable to a list"""
        ret = events
        if not isinstance(events, (tuple, list)):
            ret = [events]
        return ret

    def _event_mask(self, ds, events):
        """Select 'events' from awkward dataset 'ds'."""
<<<<<<< HEAD
        # evmask = False
=======
>>>>>>> 7afc51a (add option to provide all events at once)
        evmask = np.argwhere(np.isin(np.array(ds.event), events)).ravel()

        if isinstance(ds, pd.DataFrame):
            ret = ds.iloc[evmask]
        elif isinstance(ds, ak.Array):
            ret = ds[evmask]
        else:
            raise RuntimeError()

<<<<<<< HEAD
        # for ev in events:
        #     if not ak.sum(ds.event == ev):
        #         mes = 'Event {} is not present in file {}.'
        #         raise RuntimeError(mes.format(ev, self.outpath))
        #     evmask = evmask | (ds.event==ev)
=======
>>>>>>> 7afc51a (add option to provide all events at once)
        return ret

    def _get_event_numbers(self):
        if not os.path.exists(self.outpath):
            self.store()
        ds = ak.from_parquet(self.outpath)
        return ds.event.tolist()

    def provide(self):
        print("Providing event {} data...".format(self.tag))
        if not os.path.exists(self.outpath):
            self.store()
        return ak.from_parquet(self.outpath)

    def provide_event(self, event, merge):
        """
        Provide single event, checking if it is in cache.
        The event number is dropped due to redundancy.
        """
        if event not in self.events:
            self._add_events(event)

        ret = {}
        for k in self.var.keys():
            ret[k] = self._event_mask(self.cache[k], [event]).drop(["event"], axis=1)

        if merge:
            ret = functools.reduce(
                lambda left, right: pd.concat((left, right), axis=1), list(ret.values())
            )

        return ret

    def provide_events(self, events):
<<<<<<< HEAD
        """Provide multiple events, checking if they are in cache"""
        if isinstance(events, int) and events == -1:
            events = self.ev_numbers
=======
        """
        Provide multiple events, checking if they are in cache.
        'events=-1' means all.
        """
        if isinstance(events, int) and events==-1:
            events = self.ev_numbers
            
>>>>>>> 7afc51a (add option to provide all events at once)
        if len(events) != len(set(events)):
            mes = "You provided duplicate event numbers!"
            raise ValueError(mes)

        new_events = []
        for event in events:
            if event not in self.events:
                new_events.append(event)
        self._add_events(new_events)
        ret = {}
        for k in self.var.keys():
            ret[k] = self._event_mask(self.cache[k], events)

        return ret

    def provide_random_event(self, seed=None):
        """Provide a random event"""
        return self.provide_random_events(n=1, seed=seed)

    def provide_random_events(self, n, seed=None):
        """Provide 'n' random events ('n=-1' means all). """
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)
<<<<<<< HEAD
        events = (
            self.rng.choice(self.ev_numbers, size=n, replace=False) if n != -1 else -1
        )
=======
        events = self.rng.choice(self.ev_numbers, size=n, replace=False) if n!=-1 else -1
>>>>>>> 7afc51a (add option to provide all events at once)
        return self.provide_events(events), events

    def select(self):
        with up.open(self.indata.path, array_cache="550 MB", num_workers=8) as f:
            tree = f[self.indata.tree_path]
            allvars = set([y for x in self.var.values() for y in x.values()])
            data = tree.arrays(filter_name="/" + "|".join(allvars) + "/", library="ak")
        # data[self.var.v] = data.waferv
        # data[self.newvar.vs] = -1 * data.waferv
        # data[self.newvar.c] = "#8a2be2"
        return data

    def store(self):
        print("Store event {} data...".format(self.tag))
        data = self.select()
        if os.path.exists(self.outpath):
            os.remove(self.outpath)
        ak.to_parquet(data, self.outpath)
