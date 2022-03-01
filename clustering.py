import re
import numpy as np
import pandas as pd
import h5py
from utils import calculateRoverZfromEta

def clustering(**kwargs):
    storeInSeeds  = h5py.File(kwargs['ClusteringInSeeds'], mode='r')
    storeInTC  = h5py.File(kwargs['ClusteringInTC'], mode='r')
    storeOut = pd.HDFStore(kwargs['ClusteringOut'], mode='w')

    for falgo in kwargs['FesAlgos']:
        seed_keys = [x for x in storeInSeeds.keys() if falgo in x ]
        tc_keys  = [x for x in storeInTC.keys() if falgo in x and '_tc' in x]
        assert(len(seed_keys) == len(tc_keys))

        radiusCoeffB = kwargs['CoeffB']

        # print('CoeffA: ', kwargs['CoeffA'])
        # print('CoeffB: ', kwargs['CoeffB'])
        # print('kMidRadius: ', kwargs['MidRadius'])

        for key1, key2 in zip(tc_keys, seed_keys):
            tc = storeInTC[key1]
            tc_cols = list(tc.attrs['columns'])

            projx = tc[:,2]/tc[:,4] #tc_x / tc_z
            projy = tc[:,3]/tc[:,4] #tc_y / tc_z

            # check columns via `tc.attrs['columns']`
            radiusCoeffA = np.array( [kwargs['CoeffA'][int(xi)-1] for xi in tc[:,6]] )
            minDist = radiusCoeffA + radiusCoeffB * (kwargs['MidRadius'] - np.abs(tc[:,5]))
            # print('minDist: ', minDist)
            
            seedEn, seedX, seedY = storeInSeeds[key2]

            dRs = np.array([])
            for iseed, (en, sx, sy) in enumerate(zip(seedEn, seedX, seedY)):
                dR = np.sqrt( (projx-sx)*(projx-sx) + (projy-sy)*(projy-sy) )
                # print('d: ', dR)
                if dRs.shape == (0,):
                    dRs = np.expand_dims(dR, axis=-1)
                else:
                    dRs = np.concatenate((dRs, np.expand_dims(dR, axis=-1)), axis=1)

            #checks if each seeds has at least one seed which lies below the threshold
            pass_threshold = dRs < np.expand_dims(minDist, axis=-1)
            pass_threshold = np.logical_or.reduce(pass_threshold, axis=1)

            seeds_indexes = np.argmin(dRs, axis=1)
            seeds_energies = np.array( [seedEn[xi] for xi in seeds_indexes] )
            assert(tc[:].shape[0] == seeds_energies.shape[0]) # axis 0 stands for trigger cells

            seeds_indexes  = np.expand_dims( seeds_indexes[pass_threshold], axis=-1 )
            seeds_energies = np.expand_dims( seeds_energies[pass_threshold], axis=-1 )

            tc = tc[:][pass_threshold]

            res = np.concatenate((tc, seeds_indexes, seeds_energies), axis=1)

            key = key1.replace('_tc', '_cl')
            cols = tc_cols + [ 'seed_idx', 'seed_energy']
            assert(len(cols)==res.shape[1])
            df = pd.DataFrame(res, columns=cols)

            df['cl3d_pos_x'] = df['tc_x'] * df['tc_mipPt']
            df['cl3d_pos_y'] = df['tc_y'] * df['tc_mipPt']
            df['cl3d_pos_z'] = df['tc_z'] * df['tc_mipPt']

            cl3d = df.groupby(['seed_idx']).sum()[['cl3d_pos_x', 'cl3d_pos_y', 'cl3d_pos_z', 'tc_mipPt', 'tc_pt']]
            cl3d.rename(columns={'cl3d_pos_x': 'x',
                                 'cl3d_pos_y': 'y',
                                 'cl3d_pos_z': 'z',
                                 'tc_mipPt':   'mipPt',
                                 'tc_pt':      'pt'}, inplace=True)

            cl3d = cl3d[ cl3d.pt > kwargs['PtC3dThreshold'] ]
            
            cl3d.x /= cl3d.mipPt
            cl3d.y /= cl3d.mipPt
            cl3d.z /= cl3d.mipPt

            cl3d['x2']   = cl3d.x*cl3d.x
            cl3d['y2']   = cl3d.y*cl3d.y
            cl3d['dist'] = np.sqrt(cl3d.x2 + cl3d.y2)
            #cl3d['Rz']   = cl3d.dist / np.abs(cl3d.z)
            cl3d['eta']  = np.arcsinh(cl3d.z / cl3d.dist)
            cl3d['Rz']   = calculateRoverZfromEta(cl3d.eta)
            cl3d['phi']  = np.arctan2(cl3d.y, cl3d.x)
            cl3d['en']   = cl3d.pt*np.cosh(cl3d.eta)

            event_number = re.search('{}_([0-9]{{1,7}})_tc'.format(kwargs['FesAlgos'][0]), key1)

            if not event_number:
                raise ValueError('The event number was not extracted!')
            cl3d['event'] = event_number.group(1)
            storeOut[key] = cl3d[['en', 'x', 'y', 'z', 'eta', 'phi', 'Rz']]

    storeInSeeds.close()
    storeInTC.close()
    storeOut.close()

if __name__ == "__main__":
    from airflow.airflow_dag import clustering_kwargs        
    clustering( **clustering_kwargs )
