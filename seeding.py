import re
import numpy as np
import h5py

def validation(mipPts, event, infile, outfile,
               nbinsRz, nbinsPhi):
    """
    compares all values of 2d histogram between local and CMSSW versions
    """
    flocal  = open(infile, 'w')
    fremote = open(outfile, 'r')
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
            
    flocal.close()
    fremote.close()

def seeding(**kwargs):
    storeIn  = h5py.File(kwargs['SeedingIn'],  mode='r')
    storeOut = h5py.File(kwargs['SeedingOut'], mode='w')

    for falgo in kwargs['FesAlgos']:
        keys = [x for x in storeIn.keys() if falgo in x]
        if kwargs['Debug']:
            print(keys)

        for key in keys:
            energies, weighted_x, weighted_y = storeIn[key]

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

            south = np.roll(energies, shift=1,  axis=0)[slc]
            north = np.roll(energies, shift=-1, axis=0)[slc]
            east  = np.roll(energies, shift=-1, axis=1)[slc]
            west  = np.roll(energies, shift=1,  axis=1)[slc]
            northeast = np.roll(energies, shift=(-1,-1), axis=(0,1))[slc]
            northwest = np.roll(energies, shift=(-1,1),  axis=(0,1))[slc]
            southeast = np.roll(energies, shift=(1,-1),  axis=(0,1))[slc]
            southwest = np.roll(energies, shift=(1,1),   axis=(0,1))[slc]

            energies = energies[slc]

            maxima = ( (energies > kwargs['histoThreshold'] ) &
                       (energies >= south) & (energies > north) &
                       (energies >= east) & (energies > west) &
                       (energies >= northeast) & (energies > northwest) &
                       (energies >= southeast) & (energies > southwest) )

            seeds_idx = np.nonzero(maxima)

            res = (energies[seeds_idx], weighted_x[seeds_idx], weighted_y[seeds_idx])

            assert(len(kwargs['FesAlgos'])==1)
            event_number = re.search('{}_([0-9]{{1,7}})_group'.format(kwargs['FesAlgos'][0]),
                                                                    key).group(1)

            if kwargs['Debug']:
                print('Ev:{}'.format(event_number))
                print('Seeds bins: {}'.format(seeds_idx))
                print('NSeeds={}\tMipPt={}\tX={}\tY={}\n'
                      .format(len(res[0]), res[0], res[1], res[2]) )

            storeOut[key] = res
            storeOut[key].attrs['columns'] = ['seedEn', 'seedX', 'seedY']
            storeOut[key].attrs['doc'] = 'Smoothed energies and projected bin positions of seeds'

    storeIn.close()
    storeOut.close()

if __name__ == "__main__":
    from airflow.airflow_dag import seeding_kwargs        
    seeding( **seeding_kwargs )
