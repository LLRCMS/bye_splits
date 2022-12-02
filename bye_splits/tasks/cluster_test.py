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
import itertools

from progress.bar import IncrementalBar

def cluster(pars, **kw):
    inclusteringseeds = common.fill_path(kw['ClusterInSeeds'], **pars)
    inclusteringtc = common.fill_path(kw['ClusterInTC'], **pars)
    #outclusteringvalidation = common.fill_path(kw['ClusterOutValidation'], **pars)

    with h5py.File(inclusteringseeds, mode='r') as storeInSeeds, h5py.File(inclusteringtc, mode='r') as storeInTC:
        # Note that the 498 events in storeInSeeds and storeInTC are a subset of the gen_cl3d_tc events
        for falgo in kw['FesAlgos']:
            seed_keys = [x for x in storeInSeeds.keys() if falgo in x  and '_group_new' in x ]
            tc_keys  = [x for x in storeInTC.keys() if falgo in x and '_tc' in x]

            assert(len(seed_keys) == len(tc_keys))

            radiusCoeffB = kw['CoeffB']
            empty_seeds = 0
            # each key is an event
            bar = IncrementalBar("Loading",max=len(tc_keys))
            for key1, key2 in zip(tc_keys, seed_keys):
                tc = storeInTC[key1]

                tc_cols = list(tc.attrs['columns'])

                radiusCoeffA = np.array( [kw['CoeffA'][int(xi)-1]
                                          for xi in tc[:, common.get_column_idx(tc_cols, 'tc_layer')]] )
                minDist = ( radiusCoeffA +
                            radiusCoeffB * (kw['MidRadius'] - np.abs(tc[:, common.get_column_idx(tc_cols, 'tc_eta_new')])) )

                seedEn, seedX, seedY = storeInSeeds[key2]

                if pars['cluster_algo'] == 'min_distance':
                    dRs = np.array([])
                    projx = tc[:, common.get_column_idx(tc_cols, 'tc_x_new')]/tc[:, common.get_column_idx(tc_cols, 'tc_z')]
                    projy = tc[:, common.get_column_idx(tc_cols, 'tc_y_new')]/tc[:, common.get_column_idx(tc_cols, 'tc_z')]

                    for iseed, (en, sx, sy) in enumerate(zip(seedEn, seedX, seedY)):
                        dR = np.sqrt( (projx-sx)*(projx-sx) + (projy-sy)*(projy-sy) )

                        if dRs.shape == (0,):
                            dRs = np.expand_dims(dR, axis=-1)
                        else:
                            dRs = np.concatenate((dRs, np.expand_dims(dR, axis=-1)),
                                                 axis=1)

                    # checks if each seed has at least one seed which lies
                    # below the threshold
                    pass_threshold = dRs < np.expand_dims(minDist, axis=-1)
                    pass_threshold = np.logical_or.reduce(pass_threshold, axis=1)

                    try:
                        seeds_indexes = np.argmin(dRs, axis=1)
                    except np.AxisError:
                        empty_seeds += 1
                        continue

                elif pars['cluster_algo'] == 'max_energy':
                    seed_max = np.argmax(seedEn)
                    seeds_indexes = np.full((tc.shape[0],), seed_max) # most energetic seed takes all
                    pass_threshold = np.ones_like(seeds_indexes, dtype=bool) # always true

                seeds_energies = np.array( [seedEn[xi] for xi in seeds_indexes] )
                # axis 0 stands for trigger cells
                assert(tc[:].shape[0] == seeds_energies.shape[0])

                seeds_indexes  = np.expand_dims( seeds_indexes[pass_threshold],
                                                 axis=-1 )
                seeds_energies = np.expand_dims( seeds_energies[pass_threshold],
                                                 axis=-1 )

                tc = tc[:][pass_threshold]

                res = np.concatenate((tc, seeds_indexes, seeds_energies), axis=1)

                key = key1.replace('_tc', '_cl')
                cols = tc_cols + [ 'seed_idx', 'seed_energy']
                assert(len(cols)==res.shape[1])
                df = pd.DataFrame(res, columns=cols)

                # df['cl3d_pos_x'] = df.tc_x * df.tc_mipPt
                # df['cl3d_pos_y'] = df.tc_y * df.tc_mipPt
                df['cl3d_pos_z'] = df.tc_z * df.tc_mipPt
                df['cl3d_pos_x_new'] = df.tc_x_new * df.tc_mipPt
                df['cl3d_pos_y_new'] = df.tc_y_new * df.tc_mipPt
                #df['num_cells'] = tc.shape[0]


                cl3d_cols = [#'cl3d_pos_x', 'cl3d_pos_y',
                             'cl3d_pos_x_new', 'cl3d_pos_y_new',
                             'cl3d_pos_z',
                             'tc_mipPt', 'tc_pt']

                cl3d = df.groupby(['seed_idx']).sum()[cl3d_cols]

                cl3d = cl3d.rename(columns={#'cl3d_pos_x'       : 'x',
                                            #'cl3d_pos_y'       : 'y',
                                            'cl3d_pos_x_new'   : 'xnew',
                                            'cl3d_pos_y_new'   : 'ynew',
                                            'cl3d_pos_z'       : 'z',
                                            'tc_mipPt'         : 'mipPt',
                                            'tc_pt'            : 'pt'})

                cl3d = cl3d[ cl3d.pt > kw['PtC3dThreshold'] ]

                cl3d.z    /= cl3d.mipPt
                cl3d.xnew /= cl3d.mipPt
                cl3d.ynew /= cl3d.mipPt

                cl3d['x2new'] = cl3d.xnew**2
                cl3d['y2new'] = cl3d.ynew**2
                cl3d['distnew'] = np.sqrt(cl3d.x2new + cl3d.y2new)
                cl3d['phinew'] = np.arctan2(cl3d.ynew, cl3d.xnew)
                cl3d['etanew'] = np.arcsinh(cl3d.z / cl3d.distnew)

                cl3d['Rz']   = common.calcRzFromEta(cl3d.etanew)
                cl3d['en']   = cl3d.pt*np.cosh(cl3d.etanew)

                #Number of TCells in cluster
                cl3d['Ncells'] = df.groupby(['seed_idx']).count()['tc_pt']

                search_str = '{}_([0-9]{{1,7}})_tc'
                search_str = search_str.format(kw['FesAlgos'][0])
                event_number = re.search(search_str, key1)

                # event_number.group(1) is still a member of the gen_cl3d_tc events
                if not event_number:
                    m = 'The event number was not extracted!'
                    raise ValueError(m)

                cl3d['event'] = event_number.group(1)
                cl3d_cols = ['en', 'xnew', 'ynew', 'z', 'Rz',
                             'etanew', 'phinew', 'Ncells']
                #storeOut[key] = cl3d[cl3d_cols]
                if key1 == tc_keys[0] and key2 == seed_keys[0]:
                    dfout = cl3d[cl3d_cols+['event']]
                else:
                    dfout = pd.concat((dfout,cl3d[cl3d_cols+['event']]), axis=0)

                bar.next()
            bar.finish()

            if kw['ForEnergy']:
                dfout.event = dfout.event.astype(int)
                coef = 'coef_'+str(kw['CoeffA'][0]).replace('.','p')
                outenergy = common.fill_path(kw['EnergyOut'], **pars)
                genlev = 'data/'+kw['File']+'.hdf5'

                # THIS MUST BE CHECKED (with __ as ___ is throwing an error)
                EnOut = pd.HDFStore(outenergy, mode='a')
                GenFile = pd.HDFStore(genlev, mode='r')

                en_df = dfout.join(GenFile[kw['FesAlgos'][0]], on='event', how='inner')

                sub_df = en_df[['event','etanew','phinew','en','cl3d_pt','genpart_exphi','genpart_exeta','genpart_energy','genpart_pt']].drop_duplicates()

                EnOut.put(coef,sub_df)
                EnOut.close()

            print('[clustering step with param={}] There were {} events without seeds.'
                  .format(pars['ipar'], empty_seeds))

    outclustering = common.fill_path(kw['ClusterOutPlot'], **pars)
    with pd.HDFStore(outclustering, mode='w') as sout:
        dfout.event = dfout.event.astype(int)
        sout['data'] = dfout

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Clustering standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')
    cluster(vars(FLAGS), **params.cluster_kw)
