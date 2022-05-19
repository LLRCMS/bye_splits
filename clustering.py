import re
import numpy as np
import pandas as pd
import h5py
from random_utils import (
    calcRzFromEta,
)
from airflow.airflow_dag import fill_path

def clustering(param, **kwargs):
    inclusteringseeds = fill_path(kwargs['ClusteringInSeeds'], param=param)
    inclusteringtc = fill_path(kwargs['ClusteringInTC'], param=param)
    outclusteringvalidation = fill_path(kwargs['ClusteringOutValidation'], param=param)
    with h5py.File(inclusteringseeds, mode='r') as storeInSeeds, h5py.File(inclusteringtc, mode='r') as storeInTC, pd.HDFStore(outclusteringvalidation, mode='w') as storeOut :

        for falgo in kwargs['FesAlgos']:
            seed_keys = [x for x in storeInSeeds.keys() if falgo in x ]
            tc_keys  = [x for x in storeInTC.keys() if falgo in x and '_tc' in x]
            assert(len(seed_keys) == len(tc_keys))
         
            radiusCoeffB = kwargs['CoeffB']
            empty_seeds = 0
            for key1, key2 in zip(tc_keys, seed_keys):
                tc = storeInTC[key1]
                tc_cols = list(tc.attrs['columns'])
         
                projx = tc[:,2]/tc[:,6] #tc_x / tc_z
                projy = tc[:,3]/tc[:,6] #tc_y / tc_z

                # check columns via `tc.attrs['columns']`
                radiusCoeffA = np.array( [kwargs['CoeffA'][int(xi)-1]
                                          for xi in tc[:,8]] )
                minDist = ( radiusCoeffA +
                           radiusCoeffB * (kwargs['MidRadius'] - np.abs(tc[:,7])) )
                # print('minDist: ', minDist)
                
                seedEn, seedX, seedY, _, _ = storeInSeeds[key2]
         
                dRs = np.array([])
                for iseed, (en, sx, sy) in enumerate(zip(seedEn, seedX, seedY)):
                    dR = np.sqrt( (projx-sx)*(projx-sx) + (projy-sy)*(projy-sy) )
         
                    if dRs.shape == (0,):
                        dRs = np.expand_dims(dR, axis=-1)
                    else:
                        dRs = np.concatenate((dRs, np.expand_dims(dR, axis=-1)),
                                             axis=1)
         
                # checks if each seeds has at least one seed which lies
                # below the threshold
                pass_threshold = dRs < np.expand_dims(minDist, axis=-1)
                pass_threshold = np.logical_or.reduce(pass_threshold, axis=1)

                try:
                    seeds_indexes = np.argmin(dRs, axis=1)
                except np.AxisError:
                    empty_seeds += 1
                    continue
                
                seeds_energies = np.array( [seedEn[xi] for xi in seeds_indexes] )
                # axis 0 stands for trigger cells
                assert(tc[:].shape[0] == seeds_energies.shape[0])
         
                seeds_indexes  = np.expand_dims( seeds_indexes[pass_threshold],
                                                axis=-1 )
                seeds_energies = np.expand_dims( seeds_energies[pass_threshold],
                                                axis=-1 )
         
                tc = tc[:][pass_threshold]
         
                res = np.concatenate((tc, seeds_indexes, seeds_energies), axis=1)
         
                key = key1.replace('_tc', '_cl')
                cols = tc_cols + [ 'seed_idx', 'seed_energy']
                assert(len(cols)==res.shape[1])
                df = pd.DataFrame(res, columns=cols)
         
                df['cl3d_pos_x'] = df.tc_x * df.tc_mipPt
                df['cl3d_pos_y'] = df.tc_y * df.tc_mipPt
                df['cl3d_pos_z'] = df.tc_z * df.tc_mipPt
                df['cl3d_pos_x_new'] = df.tc_x_new * df.tc_mipPt
                df['cl3d_pos_y_new'] = df.tc_y_new * df.tc_mipPt

                cl3d_cols = ['cl3d_pos_x', 'cl3d_pos_y',
                             'cl3d_pos_x_new', 'cl3d_pos_y_new',
                             'cl3d_pos_z',
                             'tc_mipPt', 'tc_pt']
                cl3d = df.groupby(['seed_idx']).sum()[cl3d_cols]
                cl3d = cl3d.rename(columns={'cl3d_pos_x'     : 'x',
                                            'cl3d_pos_y'     : 'y',
                                            'cl3d_pos_z'     : 'z',
                                            'cl3d_pos_x_new' : 'xnew',
                                            'cl3d_pos_y_new' : 'ynew',
                                            'tc_mipPt'       : 'mipPt',
                                            'tc_pt'          : 'pt'})
         
                cl3d = cl3d[ cl3d.pt > kwargs['PtC3dThreshold'] ]
                
                cl3d.x /= cl3d.mipPt
                cl3d.y /= cl3d.mipPt
                cl3d.z /= cl3d.mipPt
                cl3d.xnew /= cl3d.mipPt
                cl3d.ynew /= cl3d.mipPt
         
                cl3d['x2']   = cl3d.x*cl3d.x
                cl3d['y2']   = cl3d.y*cl3d.y
                cl3d['dist'] = np.sqrt(cl3d.x2 + cl3d.y2)
                cl3d['eta']  = np.arcsinh(cl3d.z / cl3d.dist)
                cl3d['Rz']   = calcRzFromEta(cl3d.eta)
                # cl3d['Rz']   = cl3d.dist / np.abs(cl3d.z)
                cl3d['phi']  = np.arctan2(cl3d.y, cl3d.x)
                cl3d['en']   = cl3d.pt*np.cosh(cl3d.eta)

                search_str = '{}_([0-9]{{1,7}})_tc'.format(kwargs['FesAlgos'][0])
                event_number = re.search(search_str, key1)
         
                if not event_number:
                    raise ValueError('The event number was not extracted!')
                
                cl3d['event'] = event_number.group(1)
                cl3d_cols = ['en', 'x', 'y', 'z', 'xnew', 'ynew', 'eta', 'phi', 'Rz']
                storeOut[key] = cl3d[cl3d_cols]
                if key1 == tc_keys[0] and key2 == seed_keys[0]:
                    dfout = cl3d[cl3d_cols+['event']]
                else:
                    dfout = pd.concat((dfout,cl3d[cl3d_cols+['event']]), axis=0)

            print('There were {} events without seeds.'.format(empty_seeds))

    outclustering = fill_path(kwargs['ClusteringOutPlot'], param=param) 
    with pd.HDFStore(outclustering, mode='w') as sout:
        dfout.event = dfout.event.astype(int)
        sout['data'] = dfout

        
if __name__ == "__main__":
    from airflow.airflow_dag import clustering_kwargs        
    clustering( **clustering_kwargs )
