import os
import random
import numpy as np
import pandas as pd
import h5py

from utils import calculateRoverZfromEta

"""
Fills split clusters information according to the Stage2 FPGA fixed binning.
"""

class SupressSettingWithCopyWarning:
    """
    Temporarily supress pandas SettingWithCopyWarning.
    It is known to ocasionally provide false positives.
    https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    """
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw

### Data Extraction ####################################################
def filling(**kwargs):
    simAlgoDFs, simAlgoFiles, simAlgoPlots = ({} for _ in range(3))
    for fe in kwargs['FesAlgos']:
        simAlgoFiles[fe] = [ kwargs['FillingIn'] ]

    for fe,files in simAlgoFiles.items():
        name = fe
        dfs = []
        for afile in files:
            with pd.HDFStore(afile, mode='r') as store:
                dfs.append(store[name])
        simAlgoDFs[fe] = pd.concat(dfs)

    simAlgoNames = sorted(simAlgoDFs.keys())
    if kwargs['Debug']:
        print('Input HDF5 keys:')
        print(simAlgoNames)

    ### Data Processing ######################################################
    enrescuts = [-0.35]

    assert(len(enrescuts)==len(kwargs['FesAlgos']))
    for i,(fe,cut) in enumerate(zip(kwargs['FesAlgos'],enrescuts)):
        df = simAlgoDFs[fe]

        if kwargs['Debug']:
            print('Cluster level information:')
            print(df.filter(regex='cl3d_*.'))

        df = df[ (df['genpart_exeta']>1.7) & (df['genpart_exeta']<2.8) ]
        assert( df[ df['cl3d_eta']<0 ].shape[0] == 0 )

        with SupressSettingWithCopyWarning():
            df.loc[:,'enres'] = df.loc[:,'cl3d_energy']-df.loc[:,'genpart_energy']
            df.loc[:,'enres'] /= df.loc[:,'genpart_energy']

        nansel = pd.isna(df['enres']) 
        nandf = df[nansel]
        nandf['enres'] = 1.1
        df = df[~nansel]
        df = pd.concat([df,nandf], sort=False)

        # select events with splitted clusters (enres < energy cut)
        # if an event has at least one cluster satisfying the enres condition,
        # all of its clusters are kept (this eases comparison with CMSSW)
        df.loc[:,'atLeastOne'] = df.groupby(['event']).apply(lambda grp: np.any(grp['enres'] < cut) )
        splittedClusters = df[ df['atLeastOne'] ]
        splittedClusters.drop(['atLeastOne'], axis=1)

        # random pick some events (fixing the seed for reproducibility)
        _events_remaining = list(splittedClusters.index.unique())
        if kwargs['Nevents'] == -1:
            _events_sample = random.sample(_events_remaining,
                                           len(_events_remaining))
        else:
            _events_sample = random.sample(_events_remaining, kwargs['Nevents'])
        splittedClusters = splittedClusters.loc[_events_sample]

        if kwargs['Debug']:
            print('SplitClusters Dataset: event random selection')
            print(splittedClusters)
            print(splittedClusters.columns)

        #splitting remaining data into cluster and tc to avoid tc data duplication
        _cl3d_vars = [x for x in splittedClusters.columns.to_list() if 'tc_' not in x]

        splittedClusters_3d = splittedClusters[_cl3d_vars]
        splittedClusters_3d = splittedClusters_3d.reset_index()

        #trigger cells info is repeated across clusters in the same event
        _tc_vars = [x for x in splittedClusters.columns.to_list() if 'cl3d' not in x]
        splittedClusters_tc = splittedClusters.groupby("event").head(1)[_tc_vars] #first() instead of head(1) also works

        _tc_vars = [x for x in _tc_vars if 'tc_' in x]
        splittedClusters_tc = splittedClusters_tc.explode( _tc_vars )

        for v in _tc_vars:
            splittedClusters_tc[v] = splittedClusters_tc[v].astype(np.float64)

        splittedClusters_tc['Rz'] = np.sqrt(splittedClusters_tc.tc_x*splittedClusters_tc.tc_x + splittedClusters_tc.tc_y*splittedClusters_tc.tc_y)  / abs(splittedClusters_tc.tc_z)
        splittedClusters_tc = splittedClusters_tc.reset_index()

        #pd cut returns np.nan when value lies outside the binning
        splittedClusters_tc['Rz_bin'] = pd.cut( splittedClusters_tc['Rz'], bins=kwargs['RzBinEdges'], labels=False )
        nansel = pd.isna(splittedClusters_tc['Rz_bin']) 
        splittedClusters_tc = splittedClusters_tc[~nansel]

        splittedClusters_tc['tc_phi_bin'] = pd.cut( splittedClusters_tc['tc_phi'], bins=kwargs['PhiBinEdges'], labels=False )
        nansel = pd.isna(splittedClusters_tc['tc_phi_bin']) 
        splittedClusters_tc = splittedClusters_tc[~nansel]

        simAlgoPlots[fe] = (splittedClusters_3d, splittedClusters_tc)

    ### Event Processing ######################################################
    with h5py.File(kwargs['FillingOut'], mode='w') as store:

        for i,(_k,(df_3d,df_tc)) in enumerate(simAlgoPlots.items()):
            for ev in df_tc['event'].unique():
                ev_tc = df_tc[ df_tc.event == ev ]
                breakpoint()
                ev_3d = df_3d[ df_3d.event == ev ]

                _simCols_tc = ['tc_phi_bin', 'Rz_bin', 'tc_layer',
                               'tc_x', 'tc_y', 'tc_z', 'tc_eta',
                               'tc_mipPt', 'tc_pt', 
                               'genpart_exeta', 'genpart_exphi']
                ev_tc = ev_tc.filter(items=_simCols_tc)
                ev_tc['weighted_x'] = ev_tc['tc_mipPt'] * ev_tc['tc_x'] / np.abs(ev_tc['tc_z'])
                ev_tc['weighted_y'] = ev_tc['tc_mipPt'] * ev_tc['tc_y'] / np.abs(ev_tc['tc_z'])

                with SupressSettingWithCopyWarning():
                    ev_3d.loc[:,'cl3d_Roverz'] = calculateRoverZfromEta(ev_3d.loc[:,'cl3d_eta'])
                    ev_3d.loc[:,'gen_Roverz']  = calculateRoverZfromEta(ev_3d.loc[:,'genpart_exeta'])

                cl3d_pos_rz  = ev_3d['cl3d_Roverz'].unique() 
                cl3d_pos_phi = ev_3d['cl3d_phi'].unique()
                cl3d_pos_eta = ev_3d['cl3d_eta'].unique()
                cl3d_en      = ev_3d['cl3d_energy'].unique()

                store[str(_k) + '_' + str(ev) + '_clpos'] = (cl3d_pos_eta, cl3d_pos_phi, cl3d_pos_rz, cl3d_en)
                store[str(_k) + '_' + str(ev) + '_clpos'].attrs['columns'] = ['cl3d_eta', 'cl3d_phi', 'cl3d_rz', 'cl3d_en']
                store[str(_k) + '_' + str(ev) + '_clpos'].attrs['doc'] = 'CMSSW cluster positions.'

                gen_pos_rz, gen_pos_phi = ev_3d['gen_Roverz'].unique(), ev_3d['genpart_exphi'].unique()
                ev_3d = ev_3d.drop(['cl3d_Roverz', 'cl3d_eta', 'cl3d_phi'], axis=1)
                assert( len(gen_pos_rz) == 1 and len(gen_pos_phi) == 1 )

                gb = ev_tc.groupby(['Rz_bin', 'tc_phi_bin'], as_index=False)
                cols_to_keep = ['Rz_bin', 'tc_phi_bin', 'tc_mipPt', 'weighted_x', 'weighted_y']
                group = gb.sum()[cols_to_keep]

                group['weighted_x'] /= group['tc_mipPt']
                group['weighted_y'] /= group['tc_mipPt'] 

                store[str(_k) + '_' + str(ev) + '_group'] = group.to_numpy()
                store[str(_k) + '_' + str(ev) + '_group'].attrs['columns'] = cols_to_keep
                store[str(_k) + '_' + str(ev) + '_group'].attrs['doc'] = 'R/z vs. Phi histo Info'

                cols_to_keep = ['Rz_bin', 'tc_phi_bin',
                                'tc_x', 'tc_y', 'tc_z',
                                'tc_eta', 'tc_layer',
                                'tc_mipPt', 'tc_pt']
                ev_tc = ev_tc[cols_to_keep]
                store[str(_k) + '_' + str(ev) + '_tc'] = ev_tc.to_numpy()
                store[str(_k) + '_' + str(ev) + '_tc'].attrs['columns'] = cols_to_keep
                store[str(_k) + '_' + str(ev) + '_tc'].attrs['doc'] = 'Trigger Cells Info'

if __name__ == "__main__":
    from airflow.airflow_dag import filling_kwargs        
    filling( **filling_kwargs )
