import re
import numpy as np
import pandas as pd
import h5py
from airflow.airflow_dag import fill_path
from random_utils import get_column_idx

def validation(**kw):
    with pd.HDFStore(kw['ClusteringOutValidation'], mode='r') as storeInLocal, h5py.File(kw['FillingOut'], mode='r') as storeInCMSSW :

        for falgo in kw['FesAlgos']:
            local_keys = [x for x in storeInLocal.keys() if falgo in x]
            cmssw_keys = ( [x for x in storeInCMSSW.keys()
                            if falgo in x and '_clpos' in x] )
            assert(len(local_keys) == len(cmssw_keys))
         
            for key1, key2 in zip(local_keys, cmssw_keys):
                local = storeInLocal[key1]
                cmssw = storeInCMSSW[key2][:]
                cmssw_cols = list(cmssw.attrs['columns'])
         
                search_str = '{}_([0-9]{{1,7}})_cl'.format(kw['FesAlgos'][0])
                event_number = re.search(search_str, key1).group(1)
         
                locEtaOld = local['eta'].to_numpy()
                locPhiOld = local['phi'].to_numpy()
                locRz  = local['Rz'].to_numpy()
                locEn  = local['en'].to_numpy()
                cmsswEta = cmssw[:][get_column_idx(cmssw_cols, 'cl3d_eta')]
                cmsswPhi = cmssw[:][get_column_idx(cmssw_cols, 'cl3d_phi')]
                cmsswRz  = cmssw[:][get_column_idx(cmssw_cols, 'cl3d_Roverz')]
                cmsswEn  = cmssw[:][get_column_idx(cmssw_cols, 'cl3d_energy')]
         
                if (len(locEtaOld) != len(cmsswEta) or
                    len(locPhiOld) != len(cmsswPhi) or
                    len(locRz)  != len(cmsswRz)   or 
                    len(locEn)  != len(cmsswEn)):
                    print('Event Number: ', event_number)
                    print(local)
                    print(len(locEtaOld))
                    print(cmssw)
                    print(len(cmsswEta))
         
                errorThreshold = .5E-3
                for i in range(len(locEtaOld)):
                    if ( abs(locEtaOld[i] - cmsswEta[i]) > errorThreshold or
                         abs(locPhiOld[i] - cmsswPhi[i]) > errorThreshold or
                         abs(locRz[i]  - cmsswRz[i])  > errorThreshold  or
                         abs(locEn[i]  - cmsswEn[i])  > errorThreshold ):
                        print('Difference found in event {}!'.format(event_number))
                        print('\tEta difference: {}'.format(locEtaOld[i] - cmsswEta[i]))
                        print('\tPhi difference: {}'.format(locPhiOld[i] - cmsswPhi[i]))
                        print('\tRz difference: {}'.format(locRz[i] - cmsswRz[i]))
                        print('\tEn difference: {}'.format(locEn[i] - cmsswEn[i]))

def stats_collector(pars, debug=False, **kw):
    outclusteringvalidation = fill_path(kw['ClusteringOutValidation'], **pars)
    outfilling = fill_path(kw['FillingOut'], **pars)
    outfillingcomp = fill_path(kw['FillingOutComp'], **pars)
    with pd.HDFStore(outclusteringvalidation, mode='r') as storeInLocal, h5py.File(outfilling, mode='r') as storeInCMSSW, pd.HDFStore(outfillingcomp, mode='r') as storeGen:
        
        for falgo in kw['FesAlgos']:
            local_keys = [x for x in storeInLocal.keys() if falgo in x]
            cmssw_keys = ( [x for x in storeInCMSSW.keys()
                            if falgo in x and '_clpos' in x] )

            # the following block removes ocasional differences due to some events not having seeds
            # (check clustering.py where the np.AxisError is catched)
            # it was introduced to fix one single event (out of 5k) which for some reason had no seeds
            del1, del2 = ([] for _ in range(2))
            search_str = '{}_([0-9]{{1,7}})_cl'.format(kw['FesAlgos'][0])
            nev1 = [ re.search(search_str, k).group(1) for k in local_keys ]
            nev2 = [ re.search(search_str, k).group(1) for k in cmssw_keys ]
            diff1 = list(set(nev1) - set(nev2))
            diff2 = list(set(nev2) - set(nev1))
            for elem in local_keys:
                for d in diff1:
                    if d in elem: del1.append(elem)
            for elem in cmssw_keys:
                for d in diff2:
                    if d in elem: del2.append(elem)

            local_keys = [x for x in local_keys if x not in del1]
            cmssw_keys = [x for x in cmssw_keys if x not in del2]

            assert(len(local_keys) == len(cmssw_keys))
            
            df_gen = storeGen[falgo + '_gen']

            enres_old, enres_new   = ([] for _ in range(2))
            etares_old, etares_new = ([] for _ in range(2))
            phires_old, phires_new = ([] for _ in range(2))
            ntotal = len(local_keys)
            assert ntotal == len(cmssw_keys)

            c_loc1, c_loc2 = 0, 0
            c_cmssw1, c_cmssw2 = 0, 0
            for key1, key2 in zip(local_keys, cmssw_keys):
                local = storeInLocal[key1]
                cmssw = storeInCMSSW[key2]
                cmssw_cols = list(cmssw.attrs['columns'])
                
                locEtaNew = local['etanew'].to_numpy()
                locPhiNew = local['phinew'].to_numpy()
                locRz  = local['Rz'].to_numpy()
                locEn  = local['en'].to_numpy()

                cmsswEta  = cmssw[:][ get_column_idx(cmssw_cols, 'cl3d_eta') ]
                cmsswPhi  = cmssw[:][ get_column_idx(cmssw_cols, 'cl3d_phi') ]
                cmsswRz   = cmssw[:][ get_column_idx(cmssw_cols, 'cl3d_rz')  ]
                cmsswEn   = cmssw[:][ get_column_idx(cmssw_cols, 'cl3d_en')  ]

                event_number = re.search(search_str, key1).group(1)
                
                gen_en  = df_gen.loc[ int(event_number) ]['genpart_energy']
                gen_eta = df_gen.loc[ int(event_number) ]['genpart_exeta']
                gen_phi = df_gen.loc[ int(event_number) ]['genpart_exphi']

                #when the cluster is split we will have two rows
                if not isinstance(gen_en, float):
                    assert gen_en.iloc[1]  == gen_en.iloc[0]
                    assert gen_eta.iloc[1] == gen_eta.iloc[0]
                    assert gen_phi.iloc[1] == gen_phi.iloc[0]
                    gen_en  = gen_en.iloc[0]
                    gen_eta = gen_eta.iloc[0]
                    gen_phi = gen_phi.iloc[0]
                                        
                # ignoring the lowest energy clusters when there is a splitting
                # which give an even worse result
                _enres_old = max(cmsswEn)
                index_max_energy_cmssw = np.where(cmsswEn==_enres_old)[0][0]
                assert ( type(index_max_energy_cmssw) == np.int64 or
                         type(index_max_energy_cmssw) == np.int32 )

                _etares_old = cmsswEta[index_max_energy_cmssw]
                _phires_old = cmsswPhi[index_max_energy_cmssw]
                
                _enres_new = max(locEn)
                index_max_energy_local = np.where(locEn==_enres_new)[0][0]

                assert ( type(index_max_energy_local) == np.int64 or
                         type(index_max_energy_local) == np.int32 )
                _etares_new = locEtaNew[index_max_energy_local]
                _phires_new = locPhiNew[index_max_energy_local]

                enres_old.append ( _enres_old / gen_en )
                enres_new.append ( _enres_new / gen_en )
                etares_old.append( _etares_old - gen_eta )
                etares_new.append( _etares_new - gen_eta )
                phires_old.append( _phires_old - gen_phi )
                phires_new.append( _phires_new - gen_phi )

                if len(locEtaNew) == 1:
                    c_loc1 += 1
                elif len(locEtaNew) == 2:
                    c_loc2 += 1
                else:
                    print('Suprise')
                    raise ValueError()

                if len(cmsswEta) == 1:
                    c_cmssw1 += 1
                elif len(cmsswEta) == 2:
                    c_cmssw2 += 1
                else:
                    print('Suprise')
                    raise ValueError()

            locrat1 = float(c_loc1)/ntotal
            locrat2 = float(c_loc2)/ntotal
            cmsswrat1 = float(c_cmssw1)/ntotal
            cmsswrat2 = float(c_cmssw2)/ntotal

            if debug:
                print()
                print('CMMSW Ratio 1: {} ({})'.format(cmsswrat1, c_cmssw1))
                print('CMSSW Ratio 2: {} ({})'.format(cmsswrat2, c_cmssw2))
                print('Loc Ratio 1: {} ({})'.format(locrat1, c_loc1))
                print('Loc Ratio 2: {} ({})'.format(locrat2, c_loc2))

            return ( c_loc1, c_loc2, c_cmssw1, c_cmssw2, 
                     locrat1, locrat2, cmsswrat1, cmsswrat2,
                     enres_old, enres_new,
                     etares_old, etares_new,
                     phires_old, phires_new )
