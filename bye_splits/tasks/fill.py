# coding: utf-8

_all_ = [ 'fill' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common

import random; random.seed(18)
import numpy as np
import pandas as pd
import h5py

def fill(pars, nevents, tc_map, debug=False, **kwargs):
    """
    Fills split clusters information according to the Stage2 FPGA fixed binning.
    """    
    simAlgoDFs, simAlgoFiles, simAlgoPlots = ({} for _ in range(3))
    for fe in kwargs['FesAlgos']:
        infill = common.fill_path(kwargs['FillIn'])
        simAlgoFiles[fe] = [ infill ]

    for fe,files in simAlgoFiles.items():
        name = fe
        dfs = []
        for afile in files:
            with pd.HDFStore(afile, mode='r') as store:
                dfs.append(store[name])
        simAlgoDFs[fe] = pd.concat(dfs)

    simAlgoNames = sorted(simAlgoDFs.keys())
    if debug:
        print('Input HDF5 keys:')
        print(simAlgoNames)

    ### Data Processing ######################################################
    outfillplot = common.fill_path(kwargs['FillOutPlot'], **pars)
    outfillcomp = common.fill_path(kwargs['FillOutComp'], **pars)
    with pd.HDFStore(outfillplot, mode='w') as store, pd.HDFStore(outfillcomp, mode='w') as storeComp:

        for i,fe in enumerate(kwargs['FesAlgos']):
            df = simAlgoDFs[fe]
            df = df[ (df['genpart_exeta']>1.7) & (df['genpart_exeta']<2.8) ]
            assert( df[ df['cl3d_eta']<0 ].shape[0] == 0 )
             
            with common.SupressSettingWithCopyWarning():
                df.loc[:,'enres'] = ( df.loc[:,'cl3d_energy']
                                      - df.loc[:,'genpart_energy'] )
                df.loc[:,'enres'] /= df.loc[:,'genpart_energy']
                          
            nansel = pd.isna(df['enres'])
            nandf = df[nansel]
            nandf['enres'] = 1.1
            df = df[~nansel]
            df = pd.concat([df,nandf], sort=False)

            if pars['sel'].startswith('above_eta_'):
                #1332 events survive
                df = df[ df['genpart_exeta'] > float(pars['sel'].split('above_eta_')[1]) ]

            elif pars['sel'] == 'splits_only':
                # select events with splitted clusters (enres < energy cut)
                # if an event has at least one cluster satisfying the enres condition,
                # all of its clusters are kept (this eases comparison with CMSSW)
                df.loc[:,'atLeastOne'] = ( df.groupby(['event'])
                                          .apply(lambda grp: np.any(grp['enres'] < -0.35))
                                          )
                df = df[ df['atLeastOne'] ] #214 events survive
                df = df.drop(['atLeastOne'], axis=1)

            elif pars['sel'] == 'no_splits':
                df = df[ (df['genpart_exeta'] > 2.3) & (df['genpart_exeta'] < 2.4) ]
                df.loc[:,'goodClusters'] = (df.groupby(['event'])
                                            .apply(lambda grp: np.all(grp['enres'] > -0.15)))
                df = df[ df['goodClusters'] ] #1574 events survive
                print(df)
                df = df.drop(['goodClusters'], axis=1)
            else:
                m = 'Selection {} is not supported.'.format(pars['sel'])
                raise ValueError(m)

            if debug:
                _events_all = list(df.index.unique())
            else:
                _events_remaining = list(df.index.unique())

            storeComp[fe + '_gen'] = df.filter(regex='^gen.*')
             
            df = df.drop(['matches', 'best_match', 'cl3d_layer_pt'], axis=1)
             
            # random pick some events (fixing the seed for reproducibility)
            if nevents == -1:
                _events_sample = random.sample(_events_remaining,
                                               len(_events_remaining))
            else:
                _events_sample = random.sample(_events_remaining, nevents)
             
            if debug:
                split = df.loc[_events_all]
                # events with large eta split and good resolution
                split = split.loc[(split.index == 115441) |
                                  (split.index == 130968) |
                                  (split.index == 77678) |
                                  (split.index == 8580) |
                                  (split.index == 88782) ]
                
            else:
                split = df.loc[_events_sample]

            #splitting remaining data into cluster and tc to avoid tc data duplication
            _cl3d_vars = [x for x in split.columns.to_list()
                          if 'tc_' not in x]
             
            split_3d = split[_cl3d_vars]
            split_3d = split_3d.reset_index()
             
            #trigger cells info is repeated across clusters in the same event
            _tc_vars = [x for x in split.columns.to_list() if 'cl3d' not in x]
            split_tc = split.groupby('event').head(1)[_tc_vars]
            _tc_vars = [x for x in _tc_vars if 'tc_' in x]
            split_tc = split_tc.explode( _tc_vars )
             
            for v in _tc_vars:
                split_tc[v] = split_tc[v].astype(np.float64)
             
            split_tc.tc_id = split_tc.tc_id.astype('uint32')

            split_tc['R'] = np.sqrt(split_tc.tc_x**2 + split_tc.tc_y**2)
            split_tc['Rz'] = split_tc.R / abs(split_tc.tc_z)

            split_tc = split_tc.reset_index()
             
            #pd cut returns np.nan when value lies outside the binning
            split_tc['Rz_bin'] = pd.cut(split_tc['Rz'],
                                        bins=kwargs['RzBinEdges'],
                                        labels=False)
            nansel = pd.isna(split_tc['Rz_bin']) 
            split_tc = split_tc[~nansel]
             
            tc_map = tc_map.rename(columns={'id': 'tc_id'})

            split_tc = split_tc.merge(tc_map,
                                      on='tc_id',
                                      how='right').dropna()

            assert not np.count_nonzero(split_tc.phi_old - split_tc.tc_phi)

            split_tc['tc_x_new'] = split_tc.R * np.cos(split_tc.phi_new)
            split_tc['tc_y_new'] = split_tc.R * np.sin(split_tc.phi_new)

            split_tc['tc_eta_new'] = np.arcsinh( split_tc.tc_z /
                                                 np.sqrt(split_tc.tc_x_new**2 + split_tc.tc_y_new**2) )

            split_tc['tc_phi_bin_old'] = pd.cut(split_tc.phi_old,
                                                bins=kwargs['PhiBinEdges'],
                                                labels=False)
            split_tc['tc_phi_bin_new'] = pd.cut(split_tc.phi_new,
                                                bins=kwargs['PhiBinEdges'],
                                                labels=False)

            nansel = pd.isna(split_tc.tc_phi_bin_new) 
            split_tc = split_tc[~nansel]

            store[fe + '_3d'] = split_3d
            store[fe + '_tc'] = split_tc

            simAlgoPlots[fe] = (split_3d, split_tc)
            
    ### Event Processing ######################################################
    outfill = common.fill_path(kwargs['FillOut'], **pars)

    with h5py.File(outfill, mode='w') as store:

        for i,(_k,(df_3d,df_tc)) in enumerate(simAlgoPlots.items()):
            for ev in df_tc['event'].unique().astype('int'):
                branches  = ['cl3d_layer_pt', 'event',
                             'genpart_reachedEE', 'enres']
                ev_tc = df_tc[ df_tc.event == ev ]                
                ev_3d = df_3d[ df_3d.event == ev ]
                if debug:
                    print(ev_3d.filter(items=branches))

                _simCols_tc = ['tc_phi_bin_old', 'tc_phi_bin_new',
                               'Rz_bin', 'tc_layer',
                               'tc_x', 'tc_y',
                               'tc_x_new', 'tc_y_new',
                               'tc_eta_new',
                               'phi_old',
                               'phi_new',
                               'tc_z', 'tc_eta', 'tc_phi',
                               'tc_mipPt', 'tc_pt', 
                               'genpart_exeta', 'genpart_exphi']
                ev_tc = ev_tc.filter(items=_simCols_tc)
                wght_f = lambda pos: ev_tc.tc_mipPt*pos/np.abs(ev_tc.tc_z)
                ev_tc['wght_x'] = wght_f(ev_tc.tc_x)
                ev_tc['wght_y'] = wght_f(ev_tc.tc_y)
                
                with common.SupressSettingWithCopyWarning():
                    ev_3d['cl3d_Roverz'] = common.calcRzFromEta(ev_3d.loc[:,'cl3d_eta'])
                    ev_3d['gen_Roverz']  = common.calcRzFromEta(ev_3d.loc[:,'genpart_exeta'])

                cl3d_pos_rz  = ev_3d['cl3d_Roverz'].unique() 
                cl3d_pos_phi = ev_3d['cl3d_phi'].unique()
                cl3d_pos_eta = ev_3d['cl3d_eta'].unique()
                cl3d_en      = ev_3d['cl3d_energy'].unique()

                store[str(_k) + '_' + str(ev) + '_clpos'] = (cl3d_pos_eta, cl3d_pos_phi,
                                                             cl3d_pos_rz, cl3d_en)
                clpos_cols = ['cl3d_eta', 'cl3d_phi', 'cl3d_rz', 'cl3d_en']
                store[str(_k) + '_' + str(ev) + '_clpos'].attrs['columns'] = clpos_cols 
                store[str(_k) + '_' + str(ev) + '_clpos'].attrs['doc'] = 'CMSSW cluster positions.'

                gen_pos_rz = ev_3d['gen_Roverz'].unique()
                gen_pos_phi = ev_3d['genpart_exphi'].unique()
                ev_3d = ev_3d.drop(['cl3d_Roverz', 'cl3d_eta', 'cl3d_phi'], axis=1)
                assert( len(gen_pos_rz) == 1 and len(gen_pos_phi) == 1 )

                gb_old = ev_tc.groupby(['Rz_bin', 'tc_phi_bin_old'],
                                       as_index=False)
                gb_new = ev_tc.groupby(['Rz_bin', 'tc_phi_bin_new'],
                                       as_index=False)
                _cols_keep = ['tc_mipPt', 'wght_x', 'wght_y']
                cols_keep_old = ['Rz_bin', 'tc_phi_bin_old'] + _cols_keep
                cols_keep_new = ['Rz_bin', 'tc_phi_bin_new'] + _cols_keep

                group_old = gb_old.sum()[cols_keep_old]
                group_new = gb_new.sum()[cols_keep_new]                
                group_old.wght_x /= group_old.tc_mipPt
                group_old.wght_y /= group_old.tc_mipPt 
                group_new.wght_x /= group_new.tc_mipPt
                group_new.wght_y /= group_new.tc_mipPt 

                key_old = str(_k) + '_' + str(ev) + '_group_old'
                key_new = str(_k) + '_' + str(ev) + '_group_new'
                store[key_old] = group_old.to_numpy()
                store[key_old].attrs['columns'] = cols_keep_old
                store[key_new] = group_new.to_numpy()
                store[key_new].attrs['columns'] = cols_keep_new
                doc_m = 'R/z vs. Phi histo Info'
                store[key_old].attrs['doc'] = doc_m
                store[key_new].attrs['doc'] = doc_m

                cols_to_keep = ['Rz_bin',
                                'tc_phi_bin_old', 'tc_phi_bin_new',
                                'tc_x', 'tc_y',
                                'tc_x_new', 'tc_y_new',
                                'phi_new', 'tc_eta_new',
                                'tc_z',
                                'tc_eta', 'tc_phi',
                                'tc_layer',
                                'tc_mipPt', 'tc_pt']
                ev_tc = ev_tc[cols_to_keep]

                store[str(_k) + '_' + str(ev) + '_tc'] = ev_tc.to_numpy()
                store[str(_k) + '_' + str(ev) + '_tc'].attrs['columns'] = cols_to_keep
                store[str(_k) + '_' + str(ev) + '_tc'].attrs['doc'] = 'Trigger Cells Info'
                
                if ev == df_tc['event'].unique().astype('int')[0]:
                    group_tot_old = group_old[:]
                    group_tot_new = group_new[:]
                else:
                    group_tot_old = pd.concat((group_tot_old,
                                               group_old[:]), axis=0)
                    group_tot_new = pd.concat((group_tot_new,
                                               group_new[:]), axis=0)

    return group_tot_old, group_tot_new

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Filling standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only','no_splits') or FLAGS.sel.startswith('above_eta_')

    fill(vars(FLAGS), tc_map, **params.fill_kw)
