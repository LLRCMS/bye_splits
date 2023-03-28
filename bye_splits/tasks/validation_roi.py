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

def stats_collector_roi(pars, mode='resolution', debug=True, **kw):
    """Statistics collector used for phi bin optimization."""
    outcl  = common.fill_path(kw['ClusterOutForValidation'], **pars)
    outroi = common.fill_path(kw['ROIregionOut'], **pars)
    outgen = common.fill_path(kw['ROIclOut'], **pars)
    scl    = pd.HDFStore(outcl, mode='r')
    sroi   = h5py.File(outroi, mode='r')
    sgen   = pd.HDFStore(outgen, mode='r')

    keysroi, keyscl = sroi.keys(), scl.keys()
    assert(len(keysroi) == len(keyscl))
    ntotal = len(keyscl)

    dfgen = sgen['/df']

    data = {'resroien': [], 'resclen': [],
            'resroieta': [], 'rescleta': [],
            'resroiphi': [], 'resclphi': [],
            'roien': [], 'roieta': [], 'roiphi': [],
            'clen': [], 'cleta': [], 'clphi': [],
            'genen': [], 'geneta': [], 'genphi': [],
            'nclusters': []}

    c_cl1, c_cl2 = 0, 0
    search_str = '{}_([0-9]{{1,7}})_tc'.format(kw['FesAlgo'])
    
    for kroi, kcl in zip(keysroi, keyscl):    
        dfroi = sroi[kroi]
        dfcl = scl[kcl]
        colsroi = list(dfroi.attrs['columns'])
        roiEta  = dfroi[:, colsroi.index('tc_eta')].mean()
        roiPhi  = dfroi[:, colsroi.index('tc_phi')].mean()
        roiEn   = dfroi[:, colsroi.index('tc_energy')].sum()

        clEta = dfcl['eta'].to_numpy()
        clPhi = dfcl['phi'].to_numpy()
        clEn  = dfcl['en'].to_numpy()

        evn = re.search(search_str, kroi).group(1)
        genEvent = dfgen[dfgen.event==int(evn)]
        
        genEn  = genEvent['gen_en'].to_numpy()
        genEta = genEvent['gen_eta'].to_numpy()
        genPhi = genEvent['gen_phi'].to_numpy()
        
        #when the cluster is split we will have two rows
        if len(genEn) > 1:
            assert genEn[1]  == genEn[0]
            assert genEta[1] == genEta[0]
            assert genPhi[1] == genPhi[0]
        genEn  = genEn[0]
        genEta = genEta[0]
        genPhi = genPhi[0]
                                    
        # ignoring the lowest energy clusters when there is a splitting
        clEnMax = max(clEn)
        index_max_energy_cl = np.where(clEn==clEnMax)[0][0]
        assert ( type(index_max_energy_cl) == np.int64 or
                 type(index_max_energy_cl) == np.int32 )
        clEtaMax = clEta[index_max_energy_cl]
        clPhiMax = clPhi[index_max_energy_cl]

        data['genen'].append(genEn)
        data['geneta'].append(genEta)
        data['genphi'].append(genPhi)
        data['roien'].append(roiEn)
        data['roieta'].append(roiEta)
        data['roiphi'].append(roiPhi)
        data['clen'].append(clEnMax)
        data['cleta'].append(clEtaMax)
        data['clphi'].append(clPhiMax)
        data['resroien'].append(roiEn/genEn - 1.)
        data['resclen'].append(clEnMax/genEn - 1.)
        data['resroieta'].append(roiEta/genEta - 1.)
        data['rescleta'].append(clEtaMax/genEta - 1.)
        data['resroiphi'].append(roiEn/genPhi - 1.)
        data['resclphi'].append(clPhiMax/genPhi - 1.)
        data['nclusters'].append(len(clEn))
        # etares_old.append( etaClMax - genEta )
        # etares_new.append( etaRoiMax - genEta )
        # if phiClMax - genPhi > np.pi:
        #     phires['cl'].append(phiClMax - 2*np.pi - genPhi)
        # elif phiClMax - genPhi < -np.pi:
        #     phires['cl'].append(phiClMax + 2*np.pi - genPhi)
        # else:
        #     phires['cl'].append(phiClMax - genPhi)
                
        # if phiRoiMax - genPhi > np.pi:
        #     phires['roi'].append(phiRoiMax - 2*np.pi - genPhi)
        # elif phiRoiMax - genPhi < -np.pi:
        #     phires['roi'].append(phiRoiMax + 2*np.pi - genPhi)
        # else:
        #     phires['roi'].append(phiRoiMax - genPhi)
                        
    clrat1 = float(c_cl1) / ntotal
    clrat2 = float(c_cl2) / ntotal

    if debug:
        print()
        print('Cluster ratio singletons: {} ({})'.format(clrat1, c_cl1))
        print('Cluster ratio splits: {} ({})'.format(clrat2, c_cl2))
        
    if mode == 'resolution':
        ret = pd.DataFrame(data)

    scl.close()
    sroi.close()
    sgen.close()

    return ret

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='ROI chain validation standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()

    valid_d = params.read_task_params('valid_roi')
    stats_collector_roi(vars(FLAGS), mode='resolution', **valid_d)
