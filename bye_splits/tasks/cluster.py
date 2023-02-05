# coding: utf-8

_all_ = [ 'cluster' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits import utils
from bye_splits.utils import common

import re
import numpy as np
import pandas as pd
import h5py

def cluster(pars, **kw):
    dfout = None
    in_seeds  = common.fill_path(kw['ClusterInSeeds'], **pars)
    in_tc     = common.fill_path(kw['ClusterInTC'], **pars)
    out_valid = common.fill_path(kw['ClusterOutValidation'], **pars)
    with h5py.File(in_seeds, mode='r') as sin_seeds, h5py.File(in_tc, mode='r') as sin_tc, pd.HDFStore(out_valid, mode='w') as sout:
        seed_keys = [x for x in sin_seeds.keys() if '_group' in x ]
        tc_keys   = [x for x in sin_tc.keys() if '_tc' in x]
        assert(len(seed_keys) == len(tc_keys))
     
        radiusCoeffB = kw['CoeffB']
        empty_seeds = 0

        for tck, seedk in zip(tc_keys, seed_keys):
            tc = sin_tc[tck]
            tc_cols = list(tc.attrs['columns'])

            radiusCoeffA = np.array( [kw['CoeffA'][int(xi)-1]
                                      for xi in tc[:, common.get_column_idx(tc_cols, 'tc_layer')]] )
            minDist = (radiusCoeffA +
                       radiusCoeffB * (kw['MidRadius'] - np.abs(tc[:,common.get_column_idx(tc_cols, 'tc_eta')])))
            
            seedEn, seedX, seedY = sin_seeds[seedk]

            dRs = np.array([])
            z_tmp = tc[:, common.get_column_idx(tc_cols, 'tc_z')]
            projx = tc[:, common.get_column_idx(tc_cols, 'tc_x')] / z_tmp
            projy = tc[:, common.get_column_idx(tc_cols, 'tc_y')] / z_tmp

            for _, (en, sx, sy) in enumerate(zip(seedEn, seedX, seedY)):
                dR = np.sqrt((projx-sx)*(projx-sx) + (projy-sy)*(projy-sy))
                if dRs.shape == (0,):
                    dRs = np.expand_dims(dR, axis=-1)
                else:
                    dRs = np.concatenate((dRs, np.expand_dims(dR, axis=-1)), axis=1)
         
            # checks if each event has at least one seed laying below the threshold
            thresh = dRs < np.expand_dims(minDist, axis=-1)
            thresh = np.logical_or.reduce(thresh, axis=1)

            try:
                # assign TCs to the closest seed
                if pars['cluster_algo'] == 'min_distance':
                    seeds_indexes = np.argmin(dRs, axis=1)

                # most energetic seed takes all
                elif pars['cluster_algo'] == 'max_energy':
                    seed_max = np.argmax(seedEn)
                    seeds_indexes = np.full((tc.shape[0],), seed_max)
                    
            except np.AxisError:
                empty_seeds += 1
                continue

            seeds_energies = np.array([seedEn[xi] for xi in seeds_indexes])
            # axis 0 stands for trigger cells
            assert(tc[:].shape[0] == seeds_energies.shape[0])
     
            seeds_indexes  = np.expand_dims(seeds_indexes[thresh], axis=-1)
            seeds_energies = np.expand_dims(seeds_energies[thresh], axis=-1 )
     
            tc = tc[:][thresh]
     
            res = np.concatenate((tc, seeds_indexes, seeds_energies), axis=1)
     
            key = tck.replace('_tc', '_cl')
            cols = tc_cols + [ 'seed_idx', 'seed_energy']
            assert(len(cols)==res.shape[1])

            df = pd.DataFrame(res, columns=cols)
            df['cl3d_pos_x'] = df.tc_x * df.tc_mipPt
            df['cl3d_pos_y'] = df.tc_y * df.tc_mipPt
            df['cl3d_pos_z'] = df.tc_z * df.tc_mipPt
            
            cl3d_cols = ['cl3d_pos_x', 'cl3d_pos_y', 'cl3d_pos_z', 'tc_mipPt', 'tc_pt']
            cl3d = df.groupby(['seed_idx']).sum()[cl3d_cols]
            cl3d = cl3d.rename(columns={'cl3d_pos_x'   : 'x',
                                        'cl3d_pos_y'   : 'y',
                                        'cl3d_pos_z'   : 'z',
                                        'tc_mipPt'     : 'mipPt',
                                        'tc_pt'        : 'pt'})
            
            cl3d = cl3d[cl3d.pt > kw['PtC3dThreshold']]
            cl3d.loc[:, ['x', 'y', 'z']] = cl3d.loc[:, ['x', 'y', 'z']].div(cl3d.mipPt, axis=0)
            
            cl3d_dist = np.sqrt(cl3d.x**2 + cl3d.y**2)
            cl3d['phi'] = np.arctan2(cl3d.y, cl3d.x)
            cl3d['eta'] = np.arcsinh(cl3d.z / cl3d_dist)
            cl3d['Rz']  = common.calcRzFromEta(cl3d.eta)
            cl3d['en']  = cl3d.pt*np.cosh(cl3d.eta)

            search_str = '{}_([0-9]{{1,7}})_tc'.format(kw['FesAlgo'])
            event_number = re.search(search_str, tck)
            if not event_number:
                m = 'The event number was not extracted!'
                raise ValueError(m)
            
            cl3d['event'] = event_number.group(1)
            cl3d_cols = ['en', 'x', 'y', 'z', 'Rz', 'eta', 'phi']
            sout[key] = cl3d[cl3d_cols]
            if tck == tc_keys[0] and seedk == seed_keys[0]:
                dfout = cl3d[cl3d_cols+['event']]
            else:
                dfout = pd.concat((dfout,cl3d[cl3d_cols+['event']]), axis=0)

        print('[clustering step] There were {} events without seeds.'
              .format(empty_seeds))

    if dfout is not None:
        out = common.fill_path(kw['ClusterOutPlot'], **pars) 
        with pd.HDFStore(out, mode='w') as sout:
            dfout.event = dfout.event.astype(int)
            sout['data'] = dfout

        nevents = dfout.event.unique().shape[0]
    else:
        mes = 'No output in the cluster.'
        raise RuntimeError(mes)

    return nevents
        
if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Clustering standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    cluster(vars(FLAGS), **params.cluster_kw)
