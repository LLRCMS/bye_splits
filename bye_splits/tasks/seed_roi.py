# coding: utf-8

_all_ = [ 'seed_roi', 'dist' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common, params

import re
import yaml
import numpy as np
import h5py
import pandas as pd

def add_centrals_wafer_info(roi_ev, centr):
    roi_ev['is_central'] = 0
    iscentral = (roi_ev.tc_wu==centr[0]) & (roi_ev.tc_wv==centr[1])
    roi_ev.loc[iscentral, 'is_central'] = 1
    return roi_ev

def calc_universal_coordinates(df, varu='univ_u', varv='univ_v'):
    """
    Places all cells from different wafers into the same u/v coordinate system.
    Trims the dataframe around the central ROI wafer in the u/v space,
    according to the seeding window size.
    """
    nside = 4
    df[varu] = df.tc_cu - nside*df.tc_wu + 2*nside*df.tc_wv
    df[varv] = df.tc_cv - 2*nside*df.tc_wu + nside*df.tc_wv
    df[varu] = df[varu] - df[varu].min()
    df[varv] = df[varv] - df[varv].min()
    return df

def create_histo_uv(arr, fill, umin, umax, vmin, vmax):
    """
    Creates a 2D histogram with fixed (u,v) size.
    - arr: array where the inner axis encodes, in order:
      - 1: TC u
      - 2: TC v
      - 3: Value of interest ("counts" of the histogram, the "z axis")
    - fill stores the dummy value to fill the histogram
    - remaining variables serve to cut around the central region of the ROI
    """
    #arr_mins = arr[:,:2].min(axis=0, keepdims=True)
    #arr_bins = arr[:,:2] - arr_mins
    arr_bins_u = arr[:,0] - umin
    arr_bins_v = arr[:,1] - vmin
    
    #nu, nv = arr_bins.max(axis=0).astype(np.int32) + 1
    nu, nv = umax-umin+1, vmax-vmin+1
    
    arr_vals = arr[:,2]
    arr = np.concatenate((np.expand_dims(arr_bins_u, axis=1),
                          np.expand_dims(arr_bins_v, axis=1),
                          np.expand_dims(arr_vals, axis=1)), axis=1)
    # bin_eff = arr.shape[0] / (nu*nv)
    # print('bin efficiency: {}'.format(bin_eff))

    h = np.full((nu, nv), fill)
    for b in arr[:]:
        ubin = int(b[0])
        vbin = int(b[1])
        if ubin<0 or vbin<0 or ubin>(umax-umin) or vbin>(vmax-vmin):
            continue
        h[ubin,vbin] = b[2]
    return h

def define_histo_cuts(roi_df, centrals, wsizeu, wsizev, varu, varv):
    """
    Define u/v cuts to be applied to the histogram.
    - roi_df: dataframe containing all TCs of a ROIclOut
    - centrals: u/v of the central wafer of the ROI
    - wsize: seeding window size    
    """
    central_df = roi_df[(roi_df.tc_wu==centrals[0]) & (roi_df.tc_wv==centrals[1])][[varu,varv]]
    maxs = central_df.max() + np.array([wsizeu,wsizev])
    mins = central_df.min() - np.array([wsizeu,wsizev])
    return maxs, mins

def dist(u1, v1, u2, v2):
    """distance in an hexagonal grid"""
    s1 = u1 - v1
    s2 = u2 - v2
    return (abs(u1-u2) + abs(v1-v2) + abs(s1-s2)) / 2

def seed_roi(pars, debug=False, **kw):
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)

    uv_vars = ['univ_u', 'univ_v', 'tc_cu', 'tc_cv']
    in_tcs = common.fill_path(kw['SeedIn'], **pars)
    extra_name = '_hexdist' if cfg['seed_roi']['hexDist'] else ''
    out_seeds = common.fill_path(kw['SeedOut'] + extra_name, **pars)
    window = pars['seed_window']

    store_roi = pd.HDFStore(in_tcs, mode='r')
    store_seeds = h5py.File(out_seeds, mode='w')
    unev = [s for s in store_roi.keys() if 'central' not in s]
    for key in unev:
        inevent = store_roi[key]
        roi_uvcentrals = store_roi[key + 'central'].to_numpy()
        rois_ev = {roi_id: sub_df for roi_id, sub_df in inevent.groupby("roi_id")}

        seeds = [np.array([]) for _ in range(3)] # data storage per event
        
        for roi_ev, centrals in zip(rois_ev.values(), roi_uvcentrals):
            roi_ev = roi_ev.drop(['roi_id'], axis=1)
            roi_ev = calc_universal_coordinates(roi_ev, varu=uv_vars[0], varv=uv_vars[1])
            roi_ev = add_centrals_wafer_info(roi_ev, centrals)
            maxcuts, mincuts = define_histo_cuts(roi_ev, centrals, window, window,
                                                 varu=uv_vars[0], varv=uv_vars[1])

            # sum over layers
            group = roi_ev.groupby(by=uv_vars).sum()[['tc_mipPt', 'wght_x', 'wght_y', 'is_central']]
            group.loc[:, ['wght_x', 'wght_y']] = group.loc[:, ['wght_x', 'wght_y']].div(group.tc_mipPt, axis=0)
            group = group.reset_index()
            roi_ev = group[[uv_vars[0], uv_vars[1], 'tc_mipPt', 'wght_x', 'wght_y', 'is_central']].to_numpy()

            cuts = dict(umin=mincuts[0], umax=maxcuts[0], vmin=mincuts[1], vmax=maxcuts[1])
            energies  = create_histo_uv(roi_ev[:,[0,1,2]], fill=0., **cuts)
            wght_x    = create_histo_uv(roi_ev[:,[0,1,3]], fill=np.nan, **cuts)
            wght_y    = create_histo_uv(roi_ev[:,[0,1,4]], fill=np.nan, **cuts)
            iscentral = create_histo_uv(roi_ev[:,[0,1,5]], fill=0., **cuts)
            surroundings = []
     
            # slice to remove the included in histogram
            slc = np.index_exp[window:energies.shape[0]-window,
                               window:energies.shape[1]-window]
        
            # note: energies is by definition larger or equal to itself
            for iu in range(-window, window+1):
                for iv in range(-window, window+1):
                    if cfg['seed_roi']['hexDist'] and dist(iu,iv,0,0) > window:
                        continue
                    surroundings.append(np.roll(energies, shift=(iu,iv), axis=(0,1))[slc])

            energies  = energies[slc]
            wght_x    = wght_x[slc]
            wght_y    = wght_y[slc]
            iscentral = iscentral[slc]
     
            maxima = energies > kw['histoThreshold']
            for surr in surroundings:
                maxima = maxima & (energies >= surr)

            # iscentral > 0 indicates the u/v bin belongs to the central wafer of the ROI
            maxima = maxima & (iscentral > 0)
                
            seeds_idx = np.nonzero(maxima)
            ev_seeds = [energies[seeds_idx], wght_x[seeds_idx], wght_y[seeds_idx]]

            # join seeds in the same event from different ROIs
            for ientry in range(len(ev_seeds)):
                seeds[ientry] = np.hstack((seeds[ientry],ev_seeds[ientry]))
            
        if debug:
            print('Key: {}'.format(key))
            print('Seeds bins: {}'.format(seeds_idx))
            print('NSeeds={}\tMipPt={}\tX={}\tY={}'.format(len(res[0]),res[0],res[1],res[2])) 

        store_seeds[key] = seeds
        store_seeds[key].attrs['columns'] = ['seedEn', 'seedXdivZ', 'seedYdivZ']
        store_seeds[key].attrs['doc'] = "Seeds' energies and projected positions in {}.".format(key)

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
