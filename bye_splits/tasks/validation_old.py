# coding: utf-8

_all_ = [ 'validation', 'stats_collector' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common

import re
import numpy as np
import pandas as pd
import h5py

def validation(pars, **kw):
    in1_valid = common.fill_path(kw['ClusterOutValidation'], **pars)
    in2_valid = common.fill_path(kw['FillOut'], **pars)

    with pd.HDFStore(in1_valid, mode='r') as sloc, h5py.File(in2_valid, mode='r') as scmssw:
        local_keys = sloc.keys()
        cmssw_keys = [x for x in scmssw.keys() if '_cl' in x]
        assert(len(local_keys) == len(cmssw_keys))
     
        for key1, key2 in zip(local_keys, cmssw_keys):
            local = sloc[key1]
            cmssw = scmssw[key2]
            cmssw_cols = list(cmssw.attrs['columns'])
            cmssw = cmssw[:]
     
            search_str = '{}_([0-9]{{1,7}})_cl'.format(kw['FesAlgo'])
            event_number = re.search(search_str, key1).group(1)
     
            locEta = local['eta'].to_numpy()
            locPhi = local['phi'].to_numpy()
            locRz  = local['Rz'].to_numpy()
            locEn  = local['en'].to_numpy()
            cmsswEta = cmssw[:][cmssw_cols.index('cl3d_eta')]
            cmsswPhi = cmssw[:][cmssw_cols.index('cl3d_phi')]
            cmsswRz  = cmssw[:][cmssw_cols.index('cl3d_rz')]
            cmsswEn  = cmssw[:][cmssw_cols.index('cl3d_en')]
     
            if (len(locEta) != len(cmsswEta) or len(locPhi) != len(cmsswPhi) or
                len(locRz) != len(cmsswRz) or len(locEn) != len(cmsswEn)):
                print('Event Number: ', event_number)
                print(local, len(locEta))
                print(cmssw, len(cmsswEta))
            else:
                eThresh = 5.E-3
                for i in range(len(locEta)):
                    if (abs(locEta[i]-cmsswEta[i]) > eThresh or abs(locPhi[i]-cmsswPhi[i]) > eThresh or
                        abs(locRz[i]-cmsswRz[i]) > eThresh  or abs(locEn[i]-cmsswEn[i]) > eThresh):
                        print('Differences in event {}:'.format(event_number))
                        print('\tEta: {}'.format(locEta[i] - cmsswEta[i]))
                        print('\tPhi: {}'.format(locPhi[i] - cmsswPhi[i]))
                        print('\tRz: {}'.format(locRz[i] - cmsswRz[i]))
                        print('\tEn: {}'.format(locEn[i] - cmsswEn[i]))

def stats_collector(pars, mode='resolution', debug=True, **kw):
    """Statistics collector used for phi bin optimization."""
    out_valid    = common.fill_path(kw['ClusterOutValidation'], **pars)
    out_fill     = common.fill_path(kw['FillOut'], **pars)
    out_fillcomp = common.fill_path(kw['FillOutComp'], **pars)
    with pd.HDFStore(out_valid, mode='r') as sloc, h5py.File(out_fill, mode='r') as scmssw, pd.HDFStore(out_fillcomp, mode='r') as sgen:
        
        local_keys = sloc.keys()
        cmssw_keys = [x for x in scmssw.keys() if '_cl' in x]

        # the following block removes ocasional differences due to some events not having seeds
        # (check clustering.py where the np.AxisError is catched)
        # it was introduced to fix one single event (out of 5k) which for some reason had no seeds
        del1, del2 = ([] for _ in range(2))
        search_str = '{}_([0-9]{{1,7}})_cl'.format(kw['FesAlgo'])
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
        
        df_gen = sgen[kw['FesAlgo'] + '_gen']

        enres_old, enres_new   = ([] for _ in range(2))
        etares_old, etares_new = ([] for _ in range(2))
        phires_old, phires_new = ([] for _ in range(2))
        ntotal = len(local_keys)
        assert ntotal == len(cmssw_keys)

        c_loc1, c_loc2 = 0, 0
        c_cmssw1, c_cmssw2 = 0, 0
        evsplits = {'local': [], 'cmssw': []}

        for key1, key2 in zip(local_keys, cmssw_keys):
            local = sloc[key1]
            cmssw = scmssw[key2]
            cmssw_cols = list(cmssw.attrs['columns'])

            locEta = local['eta'].to_numpy()
            locPhi = local['phi'].to_numpy()
            locRz  = local['Rz'].to_numpy()
            locEn  = local['en'].to_numpy()

            cmsswEta  = cmssw[:][cmssw_cols.index('cl3d_eta')]
            cmsswPhi  = cmssw[:][cmssw_cols.index('cl3d_phi')]
            cmsswRz   = cmssw[:][cmssw_cols.index('cl3d_rz')]
            cmsswEn   = cmssw[:][cmssw_cols.index('cl3d_en')]

            event_number = re.search(search_str, key1).group(1)

            gen_en  = df_gen.loc[int(event_number)]['gen_en']
            gen_eta = df_gen.loc[int(event_number)]['gen_eta']
            gen_phi = df_gen.loc[int(event_number)]['gen_phi']

            #when the cluster is split we will have two rows
            if not isinstance(gen_en, (float, np.float32)):
                assert gen_en.iloc[1]  == gen_en.iloc[0]
                assert gen_eta.iloc[1] == gen_eta.iloc[0]
                assert gen_phi.iloc[1] == gen_phi.iloc[0]
                gen_en  = gen_en.iloc[0]
                gen_eta = gen_eta.iloc[0]
                gen_phi = gen_phi.iloc[0]
                                    
            # ignoring the lowest energy clusters when there is a splitting
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
            _etares_new = locEta[index_max_energy_local]
            _phires_new = locPhi[index_max_energy_local]
                
            enres_old.append ( _enres_old / gen_en )
            enres_new.append ( _enres_new / gen_en )
            etares_old.append( _etares_old - gen_eta )
            etares_new.append( _etares_new - gen_eta )
            if _phires_old - gen_phi > np.pi:
                phires_old.append(_phires_old - 2*np.pi - gen_phi)
            elif _phires_old - gen_phi < -np.pi:
                phires_old.append(_phires_old + 2*np.pi - gen_phi)
            else:
                phires_old.append(_phires_old - gen_phi)
                
            if _phires_new - gen_phi > np.pi:
                phires_new.append(_phires_new - 2*np.pi - gen_phi)
            elif _phires_new - gen_phi < -np.pi:
                phires_new.append(_phires_new + 2*np.pi - gen_phi)
            else:
                phires_new.append(_phires_new - gen_phi)

            if len(locEta) == 2:
                evsplits['local'].append(event_number)
            elif len(locEta) != 1:
                print('Suprise')
                raise ValueError()

            if len(cmsswEta) == 2:
                evsplits['cmssw'].append(event_number)
            elif len(cmsswEta) != 1:
                print('Suprise')
                raise ValueError()

        c_loc1 = ntotal-len(evsplits['local'])
        c_loc2 = len(evsplits['local'])
        c_cms1 = ntotal-len(evsplits['cmssw'])
        c_cms2 = len(evsplits['cmssw'])

        locrat1 = float(c_loc1)/ntotal
        locrat2 = float(c_loc2)/ntotal
        cmsrat1 = float(c_cms1)/ntotal
        cmsrat2 = float(c_cms2)/ntotal

        if debug:
            print()
            print('CMMSW/Custom non-split ratio:\t {}/{}'.format(cmsrat1, locrat1))
            print('CMSSW/Custom split splits:\t {}/{}'.format(cmsrat2, locrat2))

            print('CMSSW/Custom non-split count:\t {}/{}'.format(c_cms1, c_loc1))
            print('CMSSW/Custom split count:\t {}/{}'.format(c_cms2, c_loc2))

            print()
            print('List of split local events: {}'.format(evsplits['local']))
            print('List of split CMSSW events: {}'.format(evsplits['cmssw']))

        if mode == 'resolution':
            ret = pd.DataFrame({'enres_old': enres_old,
                                'enres_new': enres_new,
                                'etares_old': etares_old,
                                'etares_new': etares_new,
                                'phires_old': phires_old,  
                                'phires_new': phires_new})
        elif mode == 'ratios':
            ret = pd.DataFrame({'local1': c_loc1,
                                'local2': c_loc2,
                                'cmssw1': c_cms1,
                                'cmssw2': c_cms2,
                                'localrat1': locrat1,
                                'localrat2': locrat2,
                                'cmsswrat1': locrat1,
                                'cmsswrat2': locrat2})

        return ret

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Validation standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()

    valid_d = params.read_task_params('valid')
    validation(vars(FLAGS), **valid_d)
    stats_collector(vars(FLAGS), mode='resolution', **valid_d)
