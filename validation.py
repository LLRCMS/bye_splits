import re
import numpy as np
import pandas as pd
import h5py

def validation(**kwargs):
    with (pd.HDFStore(kwargs['ClusteringOut'], mode='r') as storeInLocal,
          h5py.File(kwargs['FillingOut'], mode='r') as storeInCMSSW):

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
                    breakpoint()
         
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
