# coding: utf-8

_all_ = [ 'smooth' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common

import numpy as np
import h5py
from copy import copy

def valid1(energies, infile, outfile, nbinsRz, nbinsPhi):
    """
    compares all values of 2d histogram between local and CMSSW versions
    """
    with open(infile, 'w') as flocal, open(outfile, 'r') as fremote :
        lines = fremote.readlines()
         
        for line in lines:
            l = line.split('\t')
            if l[0]=='\n' or '#' in l[0]:
                continue
            bin1 = int(l[0])
            bin2 = int(l[1])
            val_remote = float(l[2].replace('\n', ''))
            val_local = energies[bin1,bin2]
            if abs(val_remote-val_local)>0.001:
                print('Diff found! Bin1={}\t Bin2={}\tRemote={}\tLocal={}'.format(bin1, bin2, val_remote, val_local))
        for bin1 in range(nbinsRz):
            for bin2 in range(nbinsPhi):
                flocal.write('{}\t{}\t{}\n'.format(bin1, bin2, np.around(energies[bin1,bin2], 6)))
                

def smoothAlongRz(arr, nbinsRz, nbinsPhi):
    """
    Smoothes the energy distribution of cluster energies deposited on trigger cells.
    Works along the Rz direction ("horizontal")
    """
    weights = 2 * np.ones_like(arr)
    weights[[0,nbinsRz-1],:] = 1.5

    # add top and bottom phi rows for boundary conditions
    phiPad = np.zeros((1,nbinsPhi))
    arr = np.concatenate( (phiPad,arr,phiPad) )

    arr_new = arr + ( np.roll(arr, shift=1, axis=0) + np.roll(arr, shift=-1, axis=0) ) * 0.5

    # remove top and bottom phi rows
    arr_new = arr_new[1:arr_new.shape[0]-1]
    assert(arr_new.shape[0] == nbinsRz)
    return arr_new / weights

def smoothAlongPhi(arr, kernel,
                   binSums, nbinsRz, nbinsPhi, seedsNormByArea,
                   minROverZ, maxROverZ, areaPerTriggerCell):
    """
    Smoothes the energy distribution of cluster energies deposited on trigger cells.
    Works along the Phi direction ("horizontal")
    """
    arr_new = np.zeros_like(arr)

    nBinsSide = (np.array(binSums, dtype=np.int32) - 1) / 2; # one element per Rz bin
    assert(nBinsSide.shape[0] == nbinsRz)
    if kernel=='default':
        area = (1 + 2.0 * (1 - 0.5**nBinsSide)) # one element per Rz bin
    elif kernel=='flat_top':
        area = 5 - 2**(2-nBinsSide) # 1 + 1 + 1 + 2*(Sum[1/(2^i), {i, 1, nBinsSide - 1}])

    if seedsNormByArea:
        R1 = minROverZ + np.arange(nbinsRz) * (maxROverZ - minROverZ) / nbinsRz
        R2 = R1 + ((maxROverZ - minROverZ) / nbinsRz)
        area *= ((np.pi * (R2**2 - R1**2)) / nbinsPhi);
    else:
        #compute quantities for non-normalised-by-area histoMax
        #The 0.1 factor in bin1_10pct is an attempt to keep the same rough scale for seeds.
        #The exact value is arbitrary.
        #The value is zero, not indented, but: according to JB:
        #    the seeding threshold has been tuned with this definition,
        #    it should be kept this way until the seeding thresholds
        #    are retuned (or this 10% thingy is removed)
        bin1_10pct = int(0.1) * nbinsRz

        R1_10pct = minROverZ + bin1_10pct * (maxROverZ - minROverZ) / nbinsRz
        R2_10pct = R1_10pct + ((maxROverZ - minROverZ) / nbinsRz)
        area_10pct_ = ((np.pi * (R2_10pct**2 - R1_10pct**2)) / nbinsPhi)
        area = area * area_10pct_;

    # loop per chunk of (equal Rz) rows with a common shift to speedup
    # unfortunately np.roll's 'shift' argument must be the same for different rows
    for idx in np.unique(nBinsSide):
        roll_indices = np.where(nBinsSide == idx)[0]
        arr_copy = arr[roll_indices,:]
        arr_smooth = arr[roll_indices,:]
        for nside in range(1, int(idx)+1):
            side_sum =  np.roll(arr_copy, shift=nside,  axis=1)
            side_sum += np.roll(arr_copy, shift=-nside, axis=1)
            if kernel=='default':
                arr_smooth += side_sum / 2**nside
            elif kernel=='flat_top':
                arr_smooth += side_sum / 2**(nside-1)
        arr_new[roll_indices,:] = arr_smooth / np.expand_dims(area[roll_indices], axis=-1)

    return arr_new * areaPerTriggerCell

def createHistogram(bins, nbinsRz, nbinsPhi, fillWith):
    """
    Creates a 2D histogram with fixed (R/z vs Phi) size.
    The input event must be a 2D array where the inner axis encodes, in order:
    - 1: R/z bin index
    - 2: Phi bin index
    - 3: Value of interest ("counts" of the histogram, "z axis")
    """
    arr = np.full((nbinsRz, nbinsPhi), fillWith)

    for bin in bins[:]:
        rzbin = int(bin[0])
        phibin = int(bin[1])
        arr[rzbin,phibin] = bin[2]

    return arr

# Event by event smooth
def smooth(pars, **kwargs):
    insmooth = common.fill_path(kwargs['SmoothIn'], **pars)
    outsmooth = common.fill_path(kwargs['SmoothOut'], **pars)
    with h5py.File(insmooth,  mode='r') as storeIn, h5py.File(outsmooth, mode='w') as storeOut :

        for falgo in kwargs['FesAlgos']:
            keys_old = [x for x in storeIn.keys()
                        if falgo in x and '_group_old' in x]
            keys_new = [x for x in storeIn.keys()
                        if falgo in x and '_group_new' in x]
            
            for kold,knew in zip(keys_old,keys_new):
                en_opts = dict(nbinsRz=kwargs['NbinsRz'], nbinsPhi=kwargs['NbinsPhi'], fillWith=0.)
                xy_opts = dict(nbinsRz=kwargs['NbinsRz'], nbinsPhi=kwargs['NbinsPhi'],
                               fillWith=kwargs['Placeholder'])
                energies_old = createHistogram(storeIn[kold][:,[0,1,2]], **en_opts)
                energies_new = createHistogram(storeIn[knew][:,[0,1,2]], **en_opts)
                wght_x_old   = createHistogram(storeIn[kold][:,[0,1,3]], **xy_opts)
                wght_y_old   = createHistogram(storeIn[kold][:,[0,1,4]], **xy_opts)
                wght_x_new   = createHistogram(storeIn[knew][:,[0,1,3]], **xy_opts)
                wght_y_new   = createHistogram(storeIn[knew][:,[0,1,4]], **xy_opts)

                phi_opt = dict(binSums=kwargs['BinSums'],
                               nbinsRz=kwargs['NbinsRz'],
                               nbinsPhi=kwargs['NbinsPhi'],
                               seedsNormByArea=kwargs['SeedsNormByArea'],
                               minROverZ=kwargs['MinROverZ'],
                               maxROverZ=kwargs['MaxROverZ'],
                               areaPerTriggerCell=kwargs['AreaPerTriggerCell'])
                energies_old = smoothAlongPhi(
                    arr=energies_old,
                    kernel=pars['smooth_kernel'],
                    **phi_opt
                    )
                energies_new = smoothAlongPhi(
                    arr=energies_new,
                    kernel=pars['smooth_kernel'],
                    **phi_opt,
                    )
            
                rz_opt = (kwargs['NbinsRz'], kwargs['NbinsPhi'])
                energies_old = smoothAlongRz(
                    energies_old,
                    *rz_opt,
                )
                energies_new = smoothAlongRz(
                    energies_new,
                    *rz_opt,
                    )
         
                #printHistogram(ev)
                # 'wght_x_new', 'wght_y_new'
                cols_old = [ 'energies_old', 'wght_x_old', 'wght_y_old' ] 
                cols_new = [ 'energies_new', 'wght_x_new', 'wght_y_new' ] 

                storeOut[kold] = (energies_old, wght_x_old, wght_y_old )
                storeOut[knew] = (energies_new, wght_x_new, wght_y_new )
                
                storeOut[kold].attrs['columns'] = cols_old
                storeOut[knew].attrs['columns'] = cols_new                
                doc_m = 'Energies (post-smooth) and projected bin positions'
                doc_message = doc_m
                storeOut[kold].attrs['doc'] = doc_message
                storeOut[knew].attrs['doc'] = doc_message

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Smoothing standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    smooth(vars(FLAGS), **params.smooth_kw)
