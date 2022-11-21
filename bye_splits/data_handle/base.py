# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import abc
import awkward as ak

from utils import params, common

class BaseData(abc.ABC):
    def __init__(self, inname, outname, tag):
        self.inpath = Path('/eos/user/b/bfontana/FPGAs/new_algos/') / inname
        self.outpath = Path('/eos/user/b/bfontana/FPGAs/new_algos/') / outname
        self.tag = tag
        self.dname = 'tc'
        self.var = common.dot_dict({})
        self.newvar = common.dot_dict({})

    def provide(self, reprocess=False):
        if not os.path.exists(self.outpath) or reprocess:
            self.store()
        return ak.from_parquet(self.tag + '.parquet')

    @abc.abstractmethod
    def select(self):
        raise NotImplementedError()
    
    def store(self):
        data = self.select()
        ak.to_parquet(data, self.tag + '.parquet')
        quit()

    def variables(self):
        res = self.var.copy()
        res.update(self.newvar)
        return res
