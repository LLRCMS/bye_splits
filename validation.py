import re
import numpy as np
import pandas as pd
import h5py
from airflow.airflow_dag import fill_path

def validation(**kwargs):
    with pd.HDFStore(kwargs['ClusteringOutValidation'], mode='r') as storeInLocal, h5py.File(kwargs['FillingOut'], mode='r') as storeInCMSSW :

        for falgo in kwargs['FesAlgos']:
            local_keys = [x for x in storeInLocal.keys() if falgo in x]
            cmssw_keys = ( [x for x in storeInCMSSW.keys()
                            if falgo in x and '_clpos' in x] )
            assert(len(local_keys) == len(cmssw_keys))
         
            for key1, key2 in zip(local_keys, cmssw_keys):
                local = storeInLocal[key1]
                cmssw = storeInCMSSW[key2][:]
         
                search_str = '{}_([0-9]{{1,7}})_cl'.format(kwargs['FesAlgos'][0])
                event_number = re.search(search_str, key1).group(1)
         
                locEta = np.sort(local['eta'].to_numpy())
                locPhi = np.sort(local['phi'].to_numpy())
                locRz  = np.sort(local['Rz'].to_numpy())
                locEn  = np.sort(local['en'].to_numpy())
                cmsswEta = np.sort(cmssw[:][0])
                cmsswPhi = np.sort(cmssw[:][1])
                cmsswRz  = np.sort(cmssw[:][2])
                cmsswEn  = np.sort(cmssw[:][3])
         
                if (len(locEtaOld) != len(cmsswEta) or
                    len(locPhiOld) != len(cmsswPhi) or
                    len(locRz)  != len(cmsswRz)   or 
                    len(locEn)  != len(cmsswEn)):
                    print('Event Number: ', event_number)
                    print(local)
                    print(len(locEta))
                    print(cmssw)
                    print(len(cmsswEta))
         
                errorThreshold = .5E-3
                for i in range(len(locEta)):
                    if ( abs(locEtaOld[i] - cmsswEta[i]) > errorThreshold or
                         abs(locPhiOld[i] - cmsswPhi[i]) > errorThreshold or
                         abs(locRz[i]  - cmsswRz[i])  > errorThreshold  or
                         abs(locEn[i]  - cmsswEn[i])  > errorThreshold ):
                        print('Difference found in event {}!'.format(event_number))
                        print('\tEta difference: {}'.format(locEtaOld[i] - cmsswEta[i]))
                        print('\tPhi difference: {}'.format(locPhiOld[i] - cmsswPhi[i]))
                        print('\tRz difference: {}'.format(locRz[i] - cmsswRz[i]))
                        print('\tEn difference: {}'.format(locEn[i] - cmsswEn[i]))

def stats_collector(param, debug=False, **kwargs):
    outclusteringvalidation = fill_path(kwargs['ClusteringOutValidation'], param)
    outfilling = fill_path(kwargs['FillingOut'], param)
    outfillingcomp = fill_path(kwargs['FillingOutComp'], param)
    with pd.HDFStore(outclusteringvalidation, mode='r') as storeInLocal, h5py.File(outfilling, mode='r') as storeInCMSSW, pd.HDFStore(outfillingcomp, mode='r') as storeGen:
        
        for falgo in kwargs['FesAlgos']:
            local_keys = [x for x in storeInLocal.keys() if falgo in x]
            cmssw_keys = ( [x for x in storeInCMSSW.keys()
                            if falgo in x and '_clpos' in x] )

            # the following block removes ocasional differences due to some events not having seeds
            # (check clustering.py where the np.AxisError is catched)
            # it was introduced to fix one single event (out of 5k) which for some reason had no seeds
            del1, del2 = ([] for _ in range(2))
            search_str = '{}_([0-9]{{1,7}})_cl'.format(kwargs['FesAlgos'][0])
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

            enres_old, enres_new = ([] for _ in range(2))
            ntotal = len(local_keys)
            assert ntotal == len(cmssw_keys)

            c_loc1, c_loc2 = 0, 0
            c_rem1, c_rem2 = 0, 0
            for key1, key2 in zip(local_keys, cmssw_keys):
                local = storeInLocal[key1]
                cmssw = storeInCMSSW[key2][:]
                
                locEtaOld = np.sort(local['eta'].to_numpy())
                locPhiOld = np.sort(local['phi'].to_numpy())
                locEtaNew = np.sort(local['eta_new'].to_numpy())
                locPhiNew = np.sort(local['phi_new'].to_numpy())
                locRz     = np.sort(local['Rz'].to_numpy())
                locEn     = np.sort(local['en'].to_numpy())
                locX      = np.sort(local['x'].to_numpy())
                locY      = np.sort(local['y'].to_numpy())
                cmsswEta  = np.sort(cmssw[:][0])
                cmsswPhi  = np.sort(cmssw[:][1])
                cmsswRz   = np.sort(cmssw[:][2])
                cmsswEn   = np.sort(cmssw[:][3])

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
                assert isinstance(index_max_energy_cmssw, int)
                _etares_old = cmsswEta[index_max_energy_cmssw]
                _phires_old = cmsswPhi[index_max_energy_cmssw]
                
                _enres_new = max(locEn)
                index_max_energy_local = np.where(locEn==_enres_old)[0][0]
                assert isinstance(index_max_energy_local, int)
                _etares_new = locEtaNew[index_max_energy_local]
                _phires_new = locPhiNew[index_max_energy_local]

                enres_old.append ( _enres_old / gen_en )
                enres_new.append ( _enres_new / gen_en )
                etares_old.append( _etares_old / gen_eta )
                etares_new.append( _etares_new / gen_eta )
                phires_old.append( _phires_old / gen_phi )
                phires_new.append( _phires_new / gen_phi )

                if len(locEtaOld) == 1:
                    c_loc1 += 1
                elif len(locEtaOld) == 2:
                    c_loc2 += 1
                else:
                    print('Suprise')
                    raise ValueError()

                if len(remEta) == 1:
                    c_rem1 += 1
                elif len(remEta) == 2:
                    c_rem2 += 1
                else:
                    print('Suprise')
                    raise ValueError()

            locrat1 = float(c_loc1)/ntotal
            locrat2 = float(c_loc2)/ntotal
            remrat1 = float(c_rem1)/ntotal
            remrat2 = float(c_rem2)/ntotal

            if debug:
                print()
                print('Rem Ratio 1: {} ({})'.format(remrat1, c_rem1))
                print('Rem Ratio 2: {} ({})'.format(remrat2, c_rem2))
                print('Loc Ratio 1: {} ({})'.format(locrat1, c_loc1))
                print('Loc Ratio 2: {} ({})'.format(locrat2, c_loc2))

            return ( c_loc1, c_loc2, c_rem1, c_rem2, 
                     locrat1, locrat2, remrat1, remrat2,
                     enres_old, enres_new,
                     etares_old, etares_new,
                     phires_old, phires_new )

