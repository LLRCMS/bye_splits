# coding: utf-8

_all_ = [ 'GeometryData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import yaml
import awkward as ak
import uproot as up
import pandas as pd
import dask.dataframe as dd

from utils import params, common
from data_handle.base import BaseData

class GeometryData(BaseData):
    def __init__(self, inname=''):
        super().__init__(inname, 'geom')
        self.dname = 'tc'
        with open(params.viz_kw['CfgEventPath'], 'r') as afile:
            cfg = yaml.safe_load(afile)
            self.var = common.dot_dict(cfg['varGeometry'])

        self.readvars = list(self.var.values())
        self.readvars.remove('waferv_shift')
        self.readvars.remove('color')

    def provide(self):
        if not os.path.exists(self.outpath):
            self.store()
        return dd.read_parquet(self.outpath, engine='pyarrow')
        #return ak.from_parquet(self.outpath)

    def select(self):
        with up.open(self.inpath) as f:
            tree = f[ os.path.join('hgcaltriggergeomtester', 'TreeTriggerCells') ]
            #print(tree.show())
            data = tree.arrays(self.readvars)
            sel = (data.zside==1) & (data.subdet==1)
            fields = data[sel].fields
            for v in (self.var.side, self.var.subd):
                fields.remove(v)
            data = data[sel][fields]
            
            data = data[data.layer%2==0]
            #below is correct but much slower (no operator isin in awkward)
            #data = data[ak.Array([x in params.disconnectedTriggerLayers for x in data.layer])]
            
            #data = data.drop_duplicates(subset=[self.var.cu, self.var.cv, self.var.l])
            data[self.var.wv] = data.waferv
            data[self.var.wvs] = -1 * data.waferv
            data[self.var.c] = "#8a2be2"
            
        return data

    def store(self):
        data = self.select()
        ak.to_parquet(data, self.tag + '.parquet')
