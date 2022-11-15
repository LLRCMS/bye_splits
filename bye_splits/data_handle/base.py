# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import abc
import pandas as pd

from utils import params, common

class BaseData(abc.ABC):
    def __init__(self, inname, outname):
        # self.inpath = (Path(__file__).parent.absolute().parent.parent /
        #                params.DataFolder / inname )
        # self.outpath = (Path(__file__).parent.absolute().parent.parent /
        #                 params.DataFolder / outname )
        self.inpath = "/eos/user/b/bfontana/FPGAs/new_algos/"
        self.outpath = "/eos/user/b/bfontana/FPGAs/new_algos/"
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

    @abc.abstractmethod
    def select(self):
        raise NotImplementedError()
    
    def store(self):
        data = self.select()
        with pd.HDFStore(self.outpath, mode='w') as s:
            s[self.dname] = data

    def variables(self):
        res = self.var.copy()
        res.update(self.newvar)
        return res
