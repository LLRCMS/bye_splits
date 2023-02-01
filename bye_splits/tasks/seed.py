# coding: utf-8

_all_ = [ 'seed' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common

import re
import numpy as np
import h5py

def validation(mipPts, event, infile, outfile,
               nbinsRz, nbinsPhi):
    """
    compares all values of 2d histogram between local and CMSSW versions
    """
    with open(infile, 'w') as flocal, open(outfile, 'r') as fremote:
        lines = fremote.readlines()

        for line in lines:
            l = line.split('\t')
            if l[0]=='\n' or '#' in l[0]:
                continue
            bin1 = int(l[0])
            bin2 = int(l[1])
            val_remote = float(l[2].replace('\n', ''))
            val_local = mipPts[bin1,bin2]
            if abs(val_remote-val_local)>0.0001:
                print('Diff found! Bin1={}\t Bin2={}\tRemote={}\tLocal={}'.format(bin1, bin2, val_remote, val_local))

        for bin1 in range(nbinsRz):
            for bin2 in range(nbinsPhi):
                flocal.write('{}\t{}\t{}\n'.format(bin1, bin2, np.around(mipPts[bin1,bin2], 6)))

def seed(pars, debug=False, **kwargs):
    inseeding = common.fill_path(kwargs['SeedIn'], **pars)
    outseeding = common.fill_path(kwargs['SeedOut'], **pars)
    with h5py.File(inseeding,  mode='r') as storeIn, h5py.File(outseeding, mode='w') as storeOut:
        #Helpful for quickly verifying which events are considered
        event_list = [[int(s) for s in key.split("_") if s.isdigit()][0] for key in list(storeIn.keys())]

        for falgo in kwargs['FesAlgos']:
            keys = [x for x in storeIn.keys() if falgo in x]

            for key in keys:
                energies, wght_x, wght_y = storeIn[key]

                # if '187544' in key:
                #      validation(energies, '187544',
                #                 infile='outLocalBeforeSeeding.txt',
                #                 outfile='outCMSSWBeforeSeeding.txt',
                #                 kwargs['NbinsRz'], kwargs['NbinsPhi'] )

                # add unphysical top and bottom R/z rows for edge cases
                # fill the rows with negative (unphysical) energy values
                # boundary conditions on the phi axis are satisfied by 'np.roll'
                phiPad = -1 * np.ones((1,kwargs['NbinsPhi']))
                energies = np.concatenate( (phiPad,energies,phiPad) )

                #remove padding
                slc = slice(1,energies.shape[0]-1)

                window_size_phi = kwargs['WindowPhiDim']
                window_size_Rz  = 1
                surroundings = []

                # note: energies is by definition larger or equal to itself
                for iRz in range(-window_size_Rz, window_size_Rz+1):
                    for iphi in range(-window_size_phi, window_size_phi+1):
                        surroundings.append( np.roll(energies, shift=(iRz,iphi),
                                                     axis=(0,1))[slc] )

                # south = np.roll(energies, shift=1,  axis=0)[slc]
                # north = np.roll(energies, shift=-1, axis=0)[slc]
                # east  = np.roll(energies, shift=-1, axis=1)[slc]
                # west  = np.roll(energies, shift=1,  axis=1)[slc]
                # northeast = np.roll(energies, shift=(-1,-1), axis=(0,1))[slc]
                # northwest = np.roll(energies, shift=(-1,1),  axis=(0,1))[slc]
                # southeast = np.roll(energies, shift=(1,-1),  axis=(0,1))[slc]
                # southwest = np.roll(energies, shift=(1,1),   axis=(0,1))[slc]

                energies = energies[slc]

                # maxima = ( (energies > kwargs['histoThreshold'] ) &
                #            (energies >= south) & (energies > north) &
                #            (energies >= east) & (energies > west) &
                #            (energies >= northeast) & (energies > northwest) &
                #            (energies >= southeast) & (energies > southwest) )
                # TO DO: UPDATE THE >= WITH SOME >
                maxima = (energies > kwargs['histoThreshold'] )
                for surr in surroundings:
                    maxima = maxima & (energies >= surr)

                seeds_idx = np.nonzero(maxima)
                res = [energies[seeds_idx], wght_x[seeds_idx], wght_y[seeds_idx]]

                # The 'flat_top' kernel might create a seed in a bin without any firing TC.
                # This happens when the two phi-adjacent bins would create two (split) clusters
                # had we used a default smoothing kernel.
                # The seed position cannot threfore be defined based on firing TC.
                # We this perform the energy weighted average of the TC of the phi-adjacent bins.
                # Note: the first check avoids an error when an event has no seeds
                if res[0].shape[0]!=0 and np.isnan(res[1])[0] and np.isnan(res[2])[0]:
                    if pars['smooth_kernel'] != 'flat_top':
                        mes = 'Seeds with {} values should appear only with flat_top smoothing.'
                        raise ValueError(mes.format(kwargs['Placeholder']))
                    elif len(res[1]) > 1:
                        mes = 'Only one cluster is expected in this scenario.'
                        raise ValueError(mes)

                    lft = (seeds_idx[0][0], seeds_idx[1][0]-1)
                    rgt = (seeds_idx[0][0], seeds_idx[1][0]+1)
                    enboth = energies[lft] + energies[rgt]
                    res[1] = np.array([(wght_x[lft]*energies[lft]+wght_x[rgt]*energies[rgt])/enboth])
                    res[2] = np.array([(wght_y[lft]*energies[lft]+wght_y[rgt]*energies[rgt])/enboth])

                assert(len(kwargs['FesAlgos'])==1)
                search_str = '{}_([0-9]{{1,7}})_group'.format(kwargs['FesAlgos'][0])

                event_number = re.search(search_str, key).group(1)

                if debug:
                    print('Ev:{}'.format(event_number))
                    print('Seeds bins: {}'.format(seeds_idx))
                    # print('NSeeds={}\tMipPt={}\tX={}\tY={}\tXnew={}\tYnew={}'
                    #       .format(len(res[0]),res[0],res[1],res[2],res[3],res[4]))
                    print('NSeeds={}\tMipPt={}\tX={}\tY={}'
                          .format(len(res[0]),res[0],res[1],res[2]))

                storeOut[key] = res

                storeOut[key].attrs['columns'] = ['seedEn',
                                                  'seedX', 'seedY',
                                                  # 'seedXnew', 'seedYnew'
                                                  ]
                doc = 'Smoothed energies and projected bin positions of seeds'
                storeOut[key].attrs['doc'] = doc

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Seeding standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    seed(vars(FLAGS), **params.seed_kw)
