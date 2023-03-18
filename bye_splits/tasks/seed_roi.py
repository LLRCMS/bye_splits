# coding: utf-8

_all_ = [ 'seed_roi' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common

import re
import numpy as np
import h5py
import pandas as pd

def create_histo_uv(arr):
    """
    Creates a 2D histogram with fixed (u,v) size.
    The input event must be a 2D array where the inner axis encodes, in order:
    - 1: TC u
    - 2: TC v
    - 3: Value of interest, likely a proxy for energy ("counts" of the histogram, "z axis")
    """
    arr_mins = arr[:,:2].min(axis=0, keepdims=True)
    arr_bins = arr[:,:2] - arr_mins
    nu, nv = arr_bins.max(axis=0).astype(np.int32) + 1
    arr_vals = arr[:,2]
    arr = np.concatenate((arr_bins,np.expand_dims(arr_vals, axis=1)), axis=1)
    # bin_eff = arr.shape[0] / (nu*nv)
    # print('bin efficiency: {}'.format(bin_eff))

    h = np.full((nu, nv), 0.)
    for b in arr[:]:
        ubin = int(b[0])
        vbin = int(b[1])
        h[ubin,vbin] = b[2]

    return h, np.squeeze(arr_mins)

def seed_roi(pars, debug=False, **kw):
    in_tcs = common.fill_path(kw['SeedIn']+'_'+kw['FesAlgo'], **pars)
    out_seeds = common.fill_path(kw['SeedOut']+'_'+kw['FesAlgo'], **pars)

    window_u = kw['WindowUDim']
    window_v = kw['WindowVDim']

    store_roi = pd.HDFStore(in_tcs, mode='r')
    store_seeds = h5py.File(out_seeds, mode='w')
    unev = store_roi.keys()
    for key in unev:
        store_seeds.create_group(key)
        roi_ev = store_roi[key]

        for il in roi_ev.tc_layer.unique():
            roi_l = roi_ev[roi_ev.tc_layer==il]
            roi_l = roi_l[['tc_cu', 'tc_cv', 'tc_mipPt']].to_numpy()
            energies, mins = create_histo_uv(roi_l)

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
            res = [energies[seeds_idx], seeds_idx[0]+mins[0], seeds_idx[1]+mins[1]]

            lkey = 'l'+str(il)
            store_seeds[key].create_dataset(lkey, data=res)
            store_seeds[key][lkey].attrs['columns'] = ['seedEn', 'seedU', 'seedV']
            store_seeds[key][lkey].attrs['doc'] = 'Seed energy, cel U and cell V for layer {} in {}'.format(il, key)

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
