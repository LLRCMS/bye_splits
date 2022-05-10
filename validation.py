import re
import numpy as np
import pandas as pd
import h5py

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
                remEta = np.sort(cmssw[:][0])
                remPhi = np.sort(cmssw[:][1])
                remRz  = np.sort(cmssw[:][2])
                remEn  = np.sort(cmssw[:][3])
         
                if (len(locEta) != len(remEta) or
                    len(locPhi) != len(remPhi) or
                    len(locRz)  != len(remRz)   or 
                    len(locEn)  != len(remEn)):
                    print('Event Number: ', event_number)
                    print(local)
                    print(len(locEta))
                    print(cmssw)
                    print(len(remEta))
         
                errorThreshold = .5E-3
                for i in range(len(locEta)):
                    if ( abs(locEta[i] - remEta[i]) > errorThreshold or
                         abs(locPhi[i] - remPhi[i]) > errorThreshold or
                         abs(locRz[i]  - remRz[i])  > errorThreshold  or
                         abs(locEn[i]  - remEn[i])  > errorThreshold ):
                        print('Difference found in event {}!'.format(event_number))
                        print('\tEta difference: {}'.format(locEta[i] - remEta[i]))
                        print('\tPhi difference: {}'.format(locPhi[i] - remPhi[i]))
                        print('\tRz difference: {}'.format(locRz[i] - remRz[i]))
                        print('\tEn difference: {}'.format(locEn[i] - remEn[i]))

def stats_collector(debug=False, **kwargs):
    with pd.HDFStore(kwargs['ClusteringOutValidation'], mode='r') as storeInLocal, h5py.File(kwargs['FillingOut'], mode='r') as storeInCMSSW, pd.HDFStore(kwargs['FillingOutComp'], mode='r') as storeGen:
        
        for falgo in kwargs['FesAlgos']:
            local_keys = [x for x in storeInLocal.keys() if falgo in x]
            cmssw_keys = ( [x for x in storeInCMSSW.keys()
                            if falgo in x and '_clpos' in x] )

            df_gen = storeGen[falgo + '_gen']

            
            assert(len(local_keys) == len(cmssw_keys))

            enres_old, enres_new = ([] for _ in range(2))
            ntotal = len(local_keys)
            assert ntotal == len(cmssw_keys)

            c_loc1, c_loc2 = 0, 0
            c_rem1, c_rem2 = 0, 0
            for key1, key2 in zip(local_keys, cmssw_keys):
                local = storeInLocal[key1]
                cmssw = storeInCMSSW[key2][:]
         
                search_str = '{}_([0-9]{{1,7}})_cl'.format(kwargs['FesAlgos'][0])
                event_number = re.search(search_str, key1).group(1)
                
                locEta = np.sort(local['eta'].to_numpy())
                locPhi = np.sort(local['phi'].to_numpy())
                locRz  = np.sort(local['Rz'].to_numpy())
                locEn  = np.sort(local['en'].to_numpy())
                remEta = np.sort(cmssw[:][0])
                remPhi = np.sort(cmssw[:][1])
                remRz  = np.sort(cmssw[:][2])
                remEn  = np.sort(cmssw[:][3])

                gen_en = df_gen.loc[ int(event_number) ]['genpart_energy']
                if not isinstance(gen_en, float): #when the cluster is split we will have two rows
                    gen_en = gen_en.iloc[0]

                # ignoring the lowest energy clusters when there is a splitting
                # which give an even worse result
                _enres_old = gen_en - max(remEn)
                _enres_new = gen_en - max(locEn)              
                enres_old.append( _enres_old / gen_en )
                enres_new.append( _enres_new / gen_en )

                if len(locEta) == 1:
                    c_loc1 += 1
                elif len(locEta) == 2:
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
                     enres_old, enres_new )

