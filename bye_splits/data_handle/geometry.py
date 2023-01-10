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

from utils import params, common
from data_handle.base import BaseData

class GeometryData(BaseData):
    def __init__(self, inname='', reprocess=False):
        super().__init__(inname, 'geom', reprocess)
        self.dname = 'tc'
        with open(params.viz_kw['CfgEventPath'], 'r') as afile:
            cfg = yaml.safe_load(afile)
            self.var = common.dot_dict(cfg['varGeometry'])

        self.readvars = list(self.var.values())
        self.readvars.remove(self.var.wvs)
        self.readvars.remove(self.var.c)

    def provide(self):
        if not os.path.exists(self.outpath) or self.reprocess:
            self.store()
        print('Providing geometry data...')
        ds = ak.from_parquet(self.outpath)

        # filter some columns to reduce memory usage
        cols_to_remove = ['x', 'y', 'z', 'color']
        cols = [x for x in ds.fields if x not in cols_to_remove]
        ds = ds[cols]
        
        return ak.to_dataframe(ds)
        
    def select(self):
        with up.open(self.inpath) as f:
            tree = f[ os.path.join('hgcaltriggergeomtester', 'TreeTriggerCells') ]
            #print(tree.show())
            data = tree.arrays(self.readvars)
            sel = (data.zside==1) & (data.subdet==1)
            fields = data.fields[:]

            for v in (self.var.side, self.var.subd):
                fields.remove(v)
            data = data[sel][fields]
            data = data[data.layer%2==1]
            #below is correct but much slower (no operator isin in awkward)
            #this cut is anyways implemented in the skimmer
            #data = data[ak.Array([x not in params.disconnectedTriggerLayers for x in data.layer])]
            
            #data = data.drop_duplicates(subset=[self.var.cu, self.var.cv, self.var.l])
            data[self.var.wv] = data.waferv
            data[self.var.wvs] = -1 * data.waferv
            data[self.var.c] = "#8a2be2"

        return data

    def store(self):
        print('Store geometry data...')
        data = self.select()
        if os.path.exists(self.outpath):
            os.remove(self.outpath)
        ak.to_parquet(data, self.outpath)
