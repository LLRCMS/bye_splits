# coding: utf-8

_all_ = []

import os
from pathlib import Path
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

import abc
import yaml
import awkward as ak
import logging

from utils import params, common

class BaseData(abc.ABC):
    """Base data management class."""
    def __init__(self, tag, reprocess, logger, is_tc):
        with open(params.CfgPath, "r") as afile:
            self.cfg = yaml.safe_load(afile)

        self.tag = tag
        self.reprocess = reprocess
        self.logger = logger
        self.is_tc = is_tc

        loc = str(params.LocalStorage)
        os.makedirs(loc, exist_ok=True)

        self.outpath = os.path.join(loc, self.tag + ".parquet")
        self.var = common.dot_dict({})
        self.newvar = common.dot_dict({})

    @abc.abstractmethod
    def select(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def store(self):
        raise NotImplementedError()
