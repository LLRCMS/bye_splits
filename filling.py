import os
import random; random.seed(18)
import numpy as np
import pandas as pd
import h5py

from random_utils import (
    calcRzFromEta,
    SupressSettingWithCopyWarning,
)
from airflow.airflow_dag import fill_path

def filling(param, nevents, tc_map, selection='splits_only', debug=False, **kwargs):
    """
    Fills split clusters information according to the Stage2 FPGA fixed binning.
    """    
    simAlgoDFs, simAlgoFiles, simAlgoPlots = ({} for _ in range(3))
    for fe in kwargs['FesAlgos']:
        infilling = fill_path(kwargs['FillingIn'])
        simAlgoFiles[fe] = [ infilling ]

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
    outfillingplot = fill_path(kwargs['FillingOutPlot'], param=param, selection=selection)
    outfillingcomp = fill_path(kwargs['FillingOutComp'], param=param, selection=selection)
    with pd.HDFStore(outfillingplot, mode='w') as store, pd.HDFStore(outfillingcomp, mode='w') as storeComp:

        for i,fe in enumerate(kwargs['FesAlgos']):
            df = simAlgoDFs[fe]
            df = df[ (df['genpart_exeta']>1.7) & (df['genpart_exeta']<2.8) ]
            assert( df[ df['cl3d_eta']<0 ].shape[0] == 0 )
             
            with SupressSettingWithCopyWarning():
                df.loc[:,'enres'] = ( df.loc[:,'cl3d_energy']
                                      - df.loc[:,'genpart_energy'] )
                df.loc[:,'enres'] /= df.loc[:,'genpart_energy']
                          
            nansel = pd.isna(df['enres'])
            nandf = df[nansel]
            nandf['enres'] = 1.1
            df = df[~nansel]
            df = pd.concat([df,nandf], sort=False)

            if selection.startswith('above_eta_'):
                #1332 events survive
                df = df[ df['genpart_exeta']
                        > float(selection.split('above_eta_')[1]) ] 
            elif selection == 'splits_only':
                # select events with splitted clusters (enres < energy cut)
                # if an event has at least one cluster satisfying the enres condition,
                # all of its clusters are kept (this eases comparison with CMSSW)
                df.loc[:,'atLeastOne'] = ( df.groupby(['event'])
                                          .apply(lambda grp:
                                                 np.any(grp['enres'] < -0.35))
                                          )
                df = df[ df['atLeastOne'] ] #214 events survive
                df = df.drop(['atLeastOne'], axis=1)
            else:
                m = 'Selection {} is not supported.'.format(selection)
                raise ValueError(m)

            #_events_all = list(df.index.unique())
            _events_remaining = list(df.index.unique())

            # _events_sample_all = random.sample(_events_all, 10000)

            storeComp[fe + '_gen'] = df.filter(regex='^gen.*')
             
            df = df.drop(['matches', 'best_match', 'cl3d_layer_pt'], axis=1)
             
            # random pick some events (fixing the seed for reproducibility)
            if nevents == -1:
                _events_sample = random.sample(_events_remaining,
                                               len(_events_remaining))
            else:
                _events_sample = random.sample(_events_remaining, nevents)
             
            #split = df.loc[_events_all]
            #split = df.loc[_events_sample + _events_sample_all]
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
            split_tc['Rz_bin'] = pd.cut( split_tc['Rz'],
                                         bins=kwargs['RzBinEdges'],
                                         labels=False )
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

            split_tc['tc_phi_bin'] = pd.cut( split_tc.phi_new,
                                             bins=kwargs['PhiBinEdges'],
                                             labels=False )
            nansel = pd.isna(split_tc.tc_phi_bin) 
            split_tc = split_tc[~nansel]

            store[fe + '_3d'] = split_3d
            store[fe + '_tc'] = split_tc

            simAlgoPlots[fe] = (split_3d, split_tc)

    ### Event Processing ######################################################
    outfilling = fill_path(kwargs['FillingOut'], param=param, selection=selection)
    with h5py.File(outfilling, mode='w') as store:

        for i,(_k,(df_3d,df_tc)) in enumerate(simAlgoPlots.items()):
            for ev in df_tc['event'].unique().astype('int'):
                branches  = ['cl3d_layer_pt', 'event',
                             'genpart_reachedEE', 'enres']
                ev_tc = df_tc[ df_tc.event == ev ]                
                ev_3d = df_3d[ df_3d.event == ev ]
                if debug:
                    print(ev_3d.filter(items=branches))

                _simCols_tc = ['tc_phi_bin', 'Rz_bin', 'tc_layer',
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
                
                with SupressSettingWithCopyWarning():
                    ev_3d['cl3d_Roverz'] = calcRzFromEta(ev_3d.loc[:,'cl3d_eta'])
                    ev_3d['gen_Roverz']  = calcRzFromEta(ev_3d.loc[:,'genpart_exeta'])

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

                gb = ev_tc.groupby(['Rz_bin', 'tc_phi_bin'], as_index=False)
                cols_to_keep = ['Rz_bin', 'tc_phi_bin', 'tc_mipPt',
                                'wght_x', 'wght_y',
                                # 'wght_x_new', 'wght_y_new'
                                ]
                group = gb.sum()[cols_to_keep]
                group.wght_x       /= group.tc_mipPt
                group.wght_y       /= group.tc_mipPt 
                    
                store[str(_k) + '_' + str(ev) + '_group'] = group.to_numpy()
                store[str(_k) + '_' + str(ev) + '_group'].attrs['columns'] = cols_to_keep
                doc_m = 'R/z vs. Phi histo Info'
                store[str(_k) + '_' + str(ev) + '_group'].attrs['doc'] = doc_m

                cols_to_keep = ['Rz_bin', 'tc_phi_bin',
                                'tc_x', 'tc_y',
                                'tc_x_new', 'tc_y_new',
                                'phi_new',
                                'tc_eta_new',
                                'tc_z',
                                'tc_eta', 'tc_phi',
                                'tc_layer',
                                'tc_mipPt', 'tc_pt']
                ev_tc = ev_tc[cols_to_keep]

                store[str(_k) + '_' + str(ev) + '_tc'] = ev_tc.to_numpy()
                store[str(_k) + '_' + str(ev) + '_tc'].attrs['columns'] = cols_to_keep
                store[str(_k) + '_' + str(ev) + '_tc'].attrs['doc'] = 'Trigger Cells Info'
                
                if ev == df_tc['event'].unique().astype('int')[0]:
                    group_tot = group[:]
                else:
                    group_tot = pd.concat((group_tot,group[:]), axis=0)

    return group_tot

if __name__ == "__main__":
    from airflow.airflow_dag import filling_kwargs        
    filling( tc_map, **filling_kwargs )
