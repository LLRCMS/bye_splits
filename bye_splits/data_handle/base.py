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
    def __init__(self, inname, tag):
        self.inpath = params.viz_kw['DataPath'] / inname
        self.tag = tag
        self.outpath = self.tag + '.parquet'
        self.var = common.dot_dict({})
        self.newvar = common.dot_dict({})

    @abc.abstractmethod
    def select(self):
        raise NotImplementedError()
    
