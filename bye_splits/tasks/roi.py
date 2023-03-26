# coding: utf-8

_all_ = [ 'roi_dummy' ]

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

def create_module_sums(df_tc):
    return df_tc.groupby(['tc_wu','tc_wv']).tc_energy.sum()

def find_initial_layer(df_tc):
    """Find the first layer of the region of interest"""
    #layer_sums = tcs.groupby(['tc_layer']).tc_mipPt.sum()
    return 9 #(layer_sums.rolling(window=k).sum().shift(-k+1)).idxmax()
    
def roi(pars, df_gen, df_cl, df_tc, **kw):
    pass

def roi_dummy_calculator(tcs, k=4, threshold=20, nearby=True):
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
        ntot = cfg['geometry']['nlayersCEE'] + cfg['geometry']['nlayersCEH']
        availLayers = [x for x in range(1,ntot+1)
                       if x not in cfg["selection"]["disconnectedTriggerLayers"]]
        
    initial_layer = find_initial_layer(tcs)

    mask = (tcs.tc_layer>=initial_layer) & (tcs.tc_layer<(availLayers[availLayers.index(initial_layer)+k]))
    input_df = tcs[mask]
        
    module_sums = create_module_sums(input_df)
    module_ROI = list(module_sums[module_sums.values >= threshold].index)

    if nearby:
        selected_modules = []
        for module in module_ROI:
            nearby_modules = [(module[0]+i, module[1]+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i*j >= 0]
            skim = (module_sums.index.isin(nearby_modules)) & (module_sums.values > 0.3 * module_sums[module]) & (module_sums.values > 10)
            skim_nearby_module = list(module_sums[skim].index)
            selected_modules.extend(skim_nearby_module)
        module_ROI = selected_modules
    
    roi_df = input_df[input_df.set_index(['tc_wu','tc_wv']).index.isin(module_ROI)]

    return roi_df

def roi_calculator():
    pass

def roi(pars, df_gen, df_cl, df_tc, **kw):
    """
    Fills split clusters information according to the Stage2 FPGA fixed binning.
    """
    df_cl = data_process.baseline_selection(df_gen, df_cl, pars['sel'], **kw)
    assert(df_cl[df_cl.cl3d_eta<0].shape[0] == 0)
    df_cl.set_index('event')

    out_cl = common.fill_path(kw['ROIclOut']+'_'+kw['FesAlgo'], **pars)
    out_tc = common.fill_path(kw['ROItcOut']+'_'+kw['FesAlgo'], **pars)
    with pd.HDFStore(out_cl, mode='w') as store_cl:
        store_cl['df'] = df_cl

    ## Event-by-event processing
    store_tc = pd.HDFStore(out_tc, mode='w')
    unev = df_tc['event'].unique().astype('int')
    for ev in unev:
        ev_tc = df_tc[df_tc.event == ev]
        ev_tc = ev_tc.reset_index().drop(['entry', 'subentry', 'event'], axis=1)
        if ev_tc.empty:
            continue

        roi_tcs = roi_dummy_calculator(ev_tc) # roi_calculator()
        if roi_tcs.empty:
            continue

        keep_tc = ['tc_wu', 'tc_wv', 'tc_cu', 'tc_cv', 'tc_x', 'tc_y', 'tc_z',
                   'tc_layer', 'tc_mipPt', 'tc_pt']
        roi_tcs = roi_tcs.filter(items=keep_tc)

        divz = lambda pos: ev_tc.tc_mipPt*pos/np.abs(ev_tc.tc_z)
        roi_tcs['wght_x'] = divz(ev_tc.tc_x)
        roi_tcs['wght_y'] = divz(ev_tc.tc_y)
        store_tc['ev'+str(ev)] = roi_tcs

    nout = len(store_tc.keys())
    store_tc.close()

    print('ROI event balance: {} in, {} out.'.format(len(unev), nout))
                   
if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Create a region of interest (ROI).')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert (FLAGS.sel in ('splits_only', 'no_splits', 'all') or
            FLAGS.sel.startswith('above_eta_'))

    df_gen, df_cl, df_tc = data_process.get_data_reco_chain_start(nevents=100)
    roi_d = params.read_task_params('roi')
    roi(vars(FLAGS), df_gen, df_cl, df_tc, **roi_d)
