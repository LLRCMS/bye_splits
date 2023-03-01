# coding: utf-8

_all_ = [ 'fill' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common
from bye_splits.data_handle import data_handle

import random; random.seed(18)
import numpy as np
import pandas as pd
import h5py

def fill(pars, df_gen, df_cl, df_tc, **kw):
    """
    Fills split clusters information according to the Stage2 FPGA fixed binning.
    """
    df1 = pd.merge(left=df_gen, right=df_cl, how='inner', on='event')

    ### Data Processing ######################################################
    outfillplot = common.fill_path(kw['FillOutPlot'], **pars)
    outfillcomp = common.fill_path(kw['FillOutComp'], **pars)
    with pd.HDFStore(outfillplot, mode='w') as store, pd.HDFStore(outfillcomp, mode='w') as storeComp:

        df1 = df1[(df1.gen_eta>kw['EtaMin']) & (df1.gen_eta<kw['EtaMax'])]
        assert(df1[df1.cl3d_eta<0].shape[0] == 0)
        
        with common.SupressSettingWithCopyWarning():
            df1['enres'] = df1.cl3d_en - df1.gen_en
            df1.enres /= df1.gen_en

        nansel = pd.isna(df1['enres'])
        nandf = df1[nansel]
        nandf['enres'] = 1.1
        df1 = df1[~nansel]
        df1 = pd.concat([df1,nandf], sort=False)
        
        if pars['sel'].startswith('above_eta_'):
            df1 = df1[df.gen_eta > float(pars['sel'].split('above_eta_')[1])]

        elif pars['sel'] == 'splits_only':
            # select events with splitted clusters (enres < energy cut)
            # if an event has at least one cluster satisfying the enres condition,
            # all of its clusters are kept (this eases comparison with CMSSW)
            evgrp = df1.groupby(['event'], sort=False)
            multiplicity = evgrp.size()
            bad_res = (evgrp.apply(lambda grp: np.any(grp['enres'] < kw['EnResSplits']))).values
            bad_res_mask = np.repeat(bad_res, multiplicity.values)
            df1 = df1[bad_res_mask]
  
        elif pars['sel'] == 'no_splits':
            df1 = df1[(df1.gen_eta > kw['EtaMinStrict']) &
                      (df1.gen_eta < kw['EtaMaxStrict'])]
            evgrp = df1.groupby(['event'], sort=False)
            multiplicity = evgrp.size()
            good_res = (evgrp.apply(lambda grp: np.all(grp['enres'] > kw['EnResNoSplits']))).values
            good_res_mask = np.repeat(good_res, multiplicity.values)
            df1 = df1[good_res_mask]

        elif pars['sel'] == 'all':
            pass
        
        else:
            m = 'Selection {} is not supported.'.format(pars['sel'])
            raise ValueError(m)
        
        #df = df.drop(['matches', 'best_match', 'cl3d_layer_pt'], axis=1)
        storeComp[kw['FesAlgo'] + '_gen'] = df1.set_index('event').filter(regex='^gen.*')

        df_3d = df1[:].reset_index()

        df_tc['R'] = np.sqrt(df_tc.tc_x**2 + df_tc.tc_y**2)
        df_tc['Rz'] = df_tc.R / abs(df_tc.tc_z)
        
        # pandas 'cut' returns np.nan when value lies outside the binning
        
        rzedges = np.linspace(kw['MinROverZ'], kw['MaxROverZ'],
                              num=kw['NbinsRz']+1)
        phiedges = np.linspace(kw['MinPhi'], kw['MaxPhi'],
                               num=kw['NbinsPhi']+1)
        df_tc['Rz_bin'] = pd.cut(df_tc.Rz, bins=rzedges, labels=False)
        df_tc['tc_phi_bin'] = pd.cut(df_tc.tc_phi, bins=phiedges, labels=False)
        nansel = (pd.isna(df_tc.Rz_bin)) & (pd.isna(df_tc.tc_phi_bin))
        df_tc = df_tc[~nansel]
  
        store[kw['FesAlgo'] + '_3d'] = df_3d
        store[kw['FesAlgo'] + '_tc'] = df_tc
  
        dfs = (df_3d, df_tc)
            
    ### Event Processing ######################################################
    outfill = common.fill_path(kw['FillOut'], **pars)

    with h5py.File(outfill, mode='w') as store:
        group_tot = None
        df_3d, df_tc = dfs
        for ev in df_tc['event'].unique().astype('int'):
            ev_tc = df_tc[df_tc.event == ev]
            ev_3d = df_3d[df_3d.event == ev]
            if ev_3d.empty or ev_tc.empty:
                continue

            keep_tc = ['tc_phi_bin', 'Rz_bin', 'tc_layer', 'tc_mipPt', 'tc_pt',
                       'tc_x', 'tc_y', 'tc_z', 'tc_eta', 'tc_phi', 'gen_eta', 'gen_phi']
            ev_tc = ev_tc.filter(items=keep_tc)
            wght_f = lambda pos: ev_tc.tc_mipPt*pos/np.abs(ev_tc.tc_z)
            ev_tc['wght_x'] = wght_f(ev_tc.tc_x)
            ev_tc['wght_y'] = wght_f(ev_tc.tc_y)
  
            with common.SupressSettingWithCopyWarning():
                ev_3d['cl3d_rz'] = common.calcRzFromEta(ev_3d.loc[:,'cl3d_eta'])
                ev_3d['gen_rz']  = common.calcRzFromEta(ev_3d.loc[:,'gen_eta'])

            cl3d_rz  = ev_3d['cl3d_rz'].unique()
            cl3d_phi = ev_3d['cl3d_phi'].unique()
            cl3d_eta = ev_3d['cl3d_eta'].unique()
            cl3d_en  = ev_3d['cl3d_en'].unique()

            store_str = kw['FesAlgo'] + '_' + str(ev) + '_cl'
            cl3d_info = {'cl3d_eta': cl3d_eta, 'cl3d_phi': cl3d_phi,
                         'cl3d_rz': cl3d_rz, 'cl3d_en': cl3d_en}
            store[store_str] = list(cl3d_info.values())
            store[store_str].attrs['columns'] = list(cl3d_info.keys())
            store[store_str].attrs['doc'] = 'CMSSW cluster info.'
            
            gen_rz = ev_3d['gen_rz'].unique()
            gen_phi = ev_3d['gen_phi'].unique()
            ev_3d = ev_3d.drop(['cl3d_rz', 'cl3d_eta', 'cl3d_phi'], axis=1)
            if len(gen_rz) != 1 or len(gen_phi) != 1:
                mes = 'Impossible! Rz: {} | Phi: {}'.format(gen_rz, gen_phi)
                raise RuntimeError(mes)
            
            group = ev_tc.groupby(['Rz_bin', 'tc_phi_bin'], as_index=False)
            cols_keep = ['Rz_bin', 'tc_phi_bin', 'tc_mipPt', 'wght_x', 'wght_y']
  
            group = group.sum()[cols_keep]
            group.loc[:, ['wght_x', 'wght_y']] = group.loc[:, ['wght_x', 'wght_y']].div(group.tc_mipPt, axis=0)

            store_str = kw['FesAlgo'] + '_' + str(ev) + '_group'
            store[store_str] = group.to_numpy()
            store[store_str].attrs['columns'] = cols_keep
            store[store_str].attrs['doc'] = 'R/z vs. Phi histo Info'
  
            cols_keep = ['Rz_bin',  'tc_phi_bin', 'tc_layer', 'tc_mipPt', 'tc_pt',
                         'tc_x', 'tc_y', 'tc_z', 'tc_eta', 'tc_phi']
            ev_tc = ev_tc[cols_keep]
            if ev_tc.empty:
                continue

            store_str = kw['FesAlgo'] + '_' + str(ev) + '_tc'
            store[store_str] = ev_tc.to_numpy()
            store[store_str].attrs['columns'] = cols_keep
            store[store_str].attrs['doc'] = 'Trigger Cells Info'

            if group_tot is not None:
                group_tot = group[:]
            else:
                group_tot = pd.concat((group_tot, group[:]), axis=0)

    return group_tot

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Filling standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert (FLAGS.sel in ('splits_only', 'no_splits', 'all') or
            FLAGS.sel.startswith('above_eta_'))

    df_gen, df_cl, df_tc = data_handle.get_data_reco_chain_start(nevents=100)
    fill_d = params.read_task_params('fill')
    fill(vars(FLAGS), df_gen, df_cl, df_tc, **fill_d)
