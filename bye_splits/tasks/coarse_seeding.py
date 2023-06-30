# coding: utf-8

_all_ = [ 'cs_dummy_calculator' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common
from bye_splits.data_handle import data_process
import utils
from utils import params

import random; random.seed(18)
import numpy as np
import pandas as pd
import h5py
import yaml
from tqdm import tqdm

def create_module_sums(df_tc):
    return df_tc.groupby(['tc_wu','tc_wv']).tc_energy.sum()

def find_initial_layer(df_tc):
    """Find the first layer of the region of interest"""
    #layer_sums = tcs.groupby(['tc_layer']).tc_mipPt.sum()
    return 7 #(layer_sums.rolling(window=k).sum().shift(-k+1)).idxmax()

def get_cs_cylinder(ev_tc, cs_tcs):
    uniqueu = cs_tcs.tc_wu.unique()
    uniquev = cs_tcs.tc_wv.unique()
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
        ncee = cfg['geometry']['nlayersCEE']
    return ev_tc[(ev_tc.tc_wu.isin(uniqueu)) & (ev_tc.tc_wv.isin(uniquev)) &
                 (ev_tc.tc_layer <= ncee)]

def get_nocs(ev_tc, ev_gen, thresh):
    etagen = float(ev_gen.gen_eta[ev_gen.gen_en == ev_gen.gen_en.max()].iloc[0])
    phigen = float(ev_gen.gen_phi[ev_gen.gen_en == ev_gen.gen_en.max()].iloc[0]) 
    ev_tc['deltaR'] = utils.common.deltaR(etagen, phigen,
                                          ev_tc.tc_eta, ev_tc.tc_phi)
    ev_tc = ev_tc[ev_tc.deltaR < thresh]
    ev_tc = ev_tc.drop(['deltaR'], axis=1)
    return ev_tc

def cs(pars, df_gen, df_cl, df_tc, **kw):
    """Waiting for CS future algorithms..."""
    pass

def cs_dummy_calculator(tcs, k=4, threshold=[170, 180, 200], 
                         eta_list=[1.5, 1.9, 2.3, 3.2], nearby=False):
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
        ntot = cfg['geometry']['nlayersCEE'] + cfg['geometry']['nlayersCEH']
        availLayers = [x for x in range(1,ntot+1)
                       if x not in cfg["selection"]["disconnectedTriggerLayers"]]
        
    initial_layer = find_initial_layer(tcs)

    mask = (tcs.tc_layer>=initial_layer) & (tcs.tc_layer<(availLayers[availLayers.index(initial_layer)+k]))
    input_df = tcs[mask]
    cs_df = pd.DataFrame()
    
    module_sums = input_df.groupby(['tc_wu','tc_wv']).tc_mipPt.sum()
    eta_coord = input_df.groupby(['tc_wu','tc_wv']).tc_eta.mean()
 
    modules_CS = []
    for index in range(len(eta_list)-1):
        modules = list(module_sums[(module_sums.values >= threshold[index]) &
                                   (eta_coord.values <= eta_list[index+1])  &
                                   (eta_coord.values > eta_list[index])].index)
        modules_CS.extend(modules)

    if nearby:
        selected_modules = []
        for module in modules_CS:
            nearby_modules = [(module[0]+s, module[1]-r)
                              for s in [-1,0,1] for r in [-1,0,1] for q in [-1,0,1]
                              if s + r + q == 0]
            skimmed_modules = module_sums[module_sums.index.isin(nearby_modules)].index
            selected_modules.extend(skimmed_modules)
        modules_CS = selected_modules

    for im, module_CS in enumerate(modules_CS):
        tc_cs = input_df[input_df.set_index(['tc_wu','tc_wv']).index.isin(modules_CS)]
        with common.SupressSettingWithCopyWarning():
                tc_cs['cs_id'] = im
        cs_df = pd.concat((cs_df, tc_cs), axis=0)

    return cs_df, pd.DataFrame(modules_CS)

def cs_calculator():
    pass

def coarse_seeding(pars, df_gen, df_cl, df_tc, **kw):
    """
    Fills split clusters information according to the Stage2 FPGA fixed binning.
    """
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)

    df_cl = data_process.baseline_selection(df_gen, df_cl, pars['sel'], **kw)
    assert(df_cl[df_cl.cl3d_eta<0].shape[0] == 0)
    df_cl.set_index('event')

    # match the cluster/gen dataset with the trigger cell dataset
    df_tc = df_tc[df_tc.event.isin(df_cl.event.unique())]

    out_cl       = common.fill_path(kw['CSclOut'],       **pars)
    out_cs      = common.fill_path(kw['CStcOut'],       **pars)
    out_nocs    = common.fill_path(kw['NoCStcOut'],     **pars)
    out_cylinder = common.fill_path(kw['CScylinderOut'], **pars)
    out_all      = common.fill_path(kw['CStcAllOut'],    **pars)
    with pd.HDFStore(out_cl, mode='w') as store_cl:
        store_cl['df'] = df_cl

    ## Event-by-event processing
    store_cs      = pd.HDFStore(out_cs, mode='w')
    store_nocs    = pd.HDFStore(out_nocs, mode='w')
    store_all      = h5py.File(out_all, mode='w')
    store_cyl_h5   = h5py.File(out_cylinder, mode='w')
    unev = df_tc['event'].unique().astype('int')
    for ev in tqdm(unev):
        ev_tc = df_tc[df_tc.event == ev]
        ev_tc = ev_tc.reset_index().drop(['entry', 'subentry', 'event'], axis=1)
        if ev_tc.empty:
            continue

        cs_tcs, uvcentral = cs_dummy_calculator(ev_tc) # cs_calculator()
        if cs_tcs.empty:
            continue

        cs_keep = ['tc_wu', 'tc_wv', 'tc_cu', 'tc_cv',
                    'tc_x', 'tc_y', 'tc_z',
                    'tc_layer', 'tc_mipPt', 'tc_pt', 'tc_energy',
                    'cs_id', 'tc_eta', 'tc_phi']
        cs_tcs = cs_tcs.filter(items=cs_keep)

        divz = lambda pos: cs_tcs.tc_mipPt*pos/np.abs(cs_tcs.tc_z)
        cs_tcs['wght_x'] = divz(cs_tcs.tc_x)
        cs_tcs['wght_y'] = divz(cs_tcs.tc_y)

        keybase = kw['FesAlgo'] + '_' + str(ev) + '_'
        keycs = keybase + 'ev'
        store_cs[keycs] = cs_tcs
        store_cs[keycs].attrs['columns'] = cs_keep
        store_cs[keycs + 'central'] = uvcentral
        #store_cs[keycs + 'central'].attrs['columns'] = ['central_u', 'central_v']

        # store TCs within a cylinder in the CS
        cylinder_tcs = get_cs_cylinder(ev_tc, cs_tcs)
        cylinder_tcs = cylinder_tcs.filter(items=cs_keep)
        store_cyl_h5[keycs] = cylinder_tcs.to_numpy()
        store_cyl_h5[keycs].attrs['columns'] = cs_keep

        # store TCs within a cylinder without considering the CS
        ev_gen = df_cl[df_cl.event == ev]
        nocs_tcs = get_nocs(ev_tc, ev_gen, 9999)
        divz = lambda pos: nocs_tcs.tc_mipPt*pos/np.abs(nocs_tcs.tc_z)
        nocs_tcs = nocs_tcs.filter(items=cs_keep)
        nocs_tcs['wght_x'] = divz(nocs_tcs.tc_x)
        nocs_tcs['wght_y'] = divz(nocs_tcs.tc_y)
        store_nocs[keycs] = nocs_tcs
        
        # store all TCs
        keyall = keybase + 'tc'
        all_keep = ['tc_layer', 'tc_mipPt', 'tc_pt', 'tc_energy',
                    'tc_wu', 'tc_wv', 'tc_cu', 'tc_cv',
                    'tc_x', 'tc_y', 'tc_z', 'tc_eta', 'tc_phi']
        all_tcs = ev_tc.filter(items=all_keep)
        store_all[keyall] = all_tcs.to_numpy()
        store_all[keyall].attrs['columns'] = all_keep

    nout = int(len(store_cs.keys())/2)
    store_cs.close()
    store_nocs.close()
    store_cyl_h5.close()
    store_all.close()

    nin = len(unev)
    print('CS event balance: {} in, {} out.'.format(nin, nout))
    return nout / nin
                   
if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Create a coase seed (CS).')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert (FLAGS.sel in ('splits_only', 'no_splits', 'all') or
            FLAGS.sel.startswith('above_eta_'))

    df_gen, df_cl, df_tc = data_process.get_data_reco_chain_start(nevents=100)
    cs_d = params.read_task_params('cs')
    coarse_seeding(vars(FLAGS), df_gen, df_cl, df_tc, **cs_d)
