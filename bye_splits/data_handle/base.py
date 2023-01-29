# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import abc
import awkward as ak
import logging

from utils import params, common

class BaseData(abc.ABC):
    def __init__(self, inname, tag, reprocess, logger):
        self.inpath = os.path.join(str(params.viz_kw['DataPath']), inname)
        self.tag = tag
        self.reprocess = reprocess
        self.logger = logger

        loc = str(params.viz_kw['LocalPath'])
        os.makedirs(loc, exist_ok=True)
        self.outpath = os.path.join(loc, self.tag + '.parquet')
        self.var = common.dot_dict({})
        self.newvar = common.dot_dict({})

    @abc.abstractmethod
    def select(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def store(self):
        raise NotImplementedError()

