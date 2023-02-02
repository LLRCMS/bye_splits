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

class InputData:
    """Storage class for input strings required to access ROOT files and trees."""
    def __init__(self):
        self._path = None
        self._dir  = None
        self._tree = None

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = os.path.join(str(params.viz_kw['DataPath']), path)
         
    @property
    def adir(self):
        return self._adir

    @adir.setter
    def adir(self, adir):
        self._adir = adir

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self._tree = tree

    @property
    def tree_path(self):
        return self._dir + '/' + self._tree
        
class BaseData(abc.ABC):
    """Base data management class."""
    def __init__(self, inname, tag, reprocess, logger, is_tc):
        self.indata = InputData()
        self.indata.path = inname

        self.tag = tag
        self.reprocess = reprocess
        self.logger = logger
        self.is_tc = is_tc

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

