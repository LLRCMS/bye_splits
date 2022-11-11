# coding: utf-8

_all_ = [ 'GeometryData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import uproot as up

from utils import params
from data_handle.base import BaseData

class GeometryData(BaseData):
    def __init__(self, inname, outname):
        super().__init__(inname, outname)
        
    def select(self):
        with up.open(self.inpath) as f:
            tree = f[ os.path.join('hgcaltriggergeomtester', 'TreeTriggerCells') ]
            #print(tree.show())
            data = tree.arrays(self.var.values(), library='pd')
            sel = (data.zside==1) & (data.subdet==1)
            data = data[sel].drop([self.var.side, self.var.subd], axis=1)
            data = data.loc[~data.layer.isin(params.disconnectedTriggerLayers)]
            data = data.drop_duplicates(subset=[self.var.u, self.var.v, self.var.l])
            data[self.var.v] = data.waferv
            data[self.newvar.vs] = -1 * data.waferv
            data[self.newvar.c] = "#8a2be2"
            
        return data
