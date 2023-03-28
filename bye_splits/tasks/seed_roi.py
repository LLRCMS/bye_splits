# coding: utf-8

_all_ = [ 'seed_roi' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common, params

import re
import numpy as np
import h5py
import pandas as pd

def calc_universal_coordinates(df, varu='univ_u', varv='univ_v'):
    nside = 4
    df[varu] = df.tc_cu - nside*df.tc_wu + 2*nside*df.tc_wv
    df[varv] = df.tc_cv - 2*nside*df.tc_wu + nside*df.tc_wv
    df[varu] = df[varu] - df[varu].min()
    df[varv] = df[varv] - df[varv].min()
    return df

def create_histo_uv(arr, fill):
    """
    Creates a 2D histogram with fixed (u,v) size.
    The input event must be a 2D array where the inner axis encodes, in order:
    - 1: TC u
    - 2: TC v
    - 3: Value of interest ("counts" of the histogram, the "z axis")
    """
    arr_mins = arr[:,:2].min(axis=0, keepdims=True)
    arr_bins = arr[:,:2] - arr_mins
    nu, nv = arr_bins.max(axis=0).astype(np.int32) + 1
    arr_vals = arr[:,2]
    arr = np.concatenate((arr_bins,np.expand_dims(arr_vals, axis=1)), axis=1)
    # bin_eff = arr.shape[0] / (nu*nv)
    # print('bin efficiency: {}'.format(bin_eff))

    h = np.full((nu, nv), fill)
    for b in arr[:]:
        ubin = int(b[0])
        vbin = int(b[1])
        h[ubin,vbin] = b[2]

    return h

def seed_roi(pars, debug=False, **kw):
    uv_vars = ['univ_u', 'univ_v', 'tc_cu', 'tc_cv', 'tc_wu', 'tc_wv']
    in_tcs = common.fill_path(kw['SeedIn'], **pars)
    out_seeds = common.fill_path(kw['SeedOut'], **pars)
    window_u = pars['seed_window']
    window_v = pars['seed_window']

    store_roi = pd.HDFStore(in_tcs, mode='r')
    store_seeds = h5py.File(out_seeds, mode='w')
    unev = store_roi.keys()
    for key in unev:
        roi_ev = store_roi[key]
        roi_ev = calc_universal_coordinates(roi_ev, varu=uv_vars[0], varv=uv_vars[1])

        # sum over layers
        group = roi_ev.groupby(by=uv_vars).sum()[['tc_mipPt', 'wght_x', 'wght_y']]
        group.loc[:, ['wght_x', 'wght_y']] = group.loc[:, ['wght_x', 'wght_y']].div(group.tc_mipPt, axis=0)
        group = group.reset_index()

        roi_ev = group[['tc_cu', 'tc_cv', 'tc_mipPt', 'wght_x', 'wght_y']].to_numpy()

        energies = create_histo_uv(roi_ev[:,[0,1,2]], fill=0.)
        wght_x   = create_histo_uv(roi_ev[:,[0,1,3]], fill=np.nan)
        wght_y   = create_histo_uv(roi_ev[:,[0,1,4]], fill=np.nan)

        surroundings = []
 
        # add unphysical top and bottom u rows and v columns TCs in the edge
        # fill the rows with negative (unphysical) energy values
        # boundary conditions on the phi axis are satisfied by 'np.roll'
        energies = np.pad(energies, pad_width=(window_u, window_v),
                          constant_values=(-1,-1))
        
        # slice to remove padding
        slc = np.index_exp[window_u:energies.shape[0]-window_u,
                           window_v:energies.shape[1]-window_v]
    
        # note: energies is by definition larger or equal to itself
        for iu in range(-window_u, window_u+1):
            for iv in range(-window_v, window_v+1):
                # if iu == 2 and iv == 2: CHANGE!!!!!!!!!!!!!!!!!!!!!!!
                #     continue
                surroundings.append(np.roll(energies, shift=(iu,iv), axis=(0,1))[slc])
 
        energies = energies[slc]
 
        maxima = (energies > kw['histoThreshold'] )
        for surr in surroundings:
            maxima = maxima & (energies >= surr)

        seeds_idx = np.nonzero(maxima)
        res = [energies[seeds_idx], wght_x[seeds_idx], wght_y[seeds_idx]]

        if debug:
            print('Ev: {}'.format(event_number))
            print('Seeds bins: {}'.format(seeds_idx))
            print('NSeeds={}\tMipPt={}\tX={}\tY={}'.format(len(res[0]),res[0],res[1],res[2])) 

        store_seeds[key] = res
        store_seeds[key].attrs['columns'] = ['seedEn', 'seedXdivZ', 'seedYdivZ']
        store_seeds[key].attrs['doc'] = "Seeds' nergies and projected positions in {}.".format(key)

    nout = len(store_seeds.keys())
    store_roi.close()
    store_seeds.close()

    print('ROI seeding event balance: {} in, {} out.'.format(len(unev), nout))

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Seeding standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()

    seed_d = params.read_task_params('seed_roi')
    seed_roi(vars(FLAGS), **seed_d)
