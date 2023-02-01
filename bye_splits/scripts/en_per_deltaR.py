import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits import utils
from bye_splits import tasks
from bye_splits.utils import common
from bye_splits.tasks import cluster
from bye_splits.tasks import cluster_test

import re
import numpy as np
import pandas as pd
import h5py

import math
import itertools

def deltar(df):
    df['deta']=df['etanew']-df['genpart_exeta']
    df['dphi']=np.abs(df['phinew']-df['genpart_exphi'])
    sel=df['dphi']>np.pi
    df['dphi']-=sel*(2*np.pi)
    return(np.sqrt(df['dphi']*df['dphi']+df['deta']*df['deta']))

def matching(event):
    # If there are no clusters within the threshold, return the one with the highest energy
    if event.matches.sum()==0:
        return event.en==event.en.max()
    # If there are multiple clusters within the threshold, take the maximum energy _*of these clusters*_
    else:
        cond_a = event.matches==True
        cond_b = event.en==event[cond_a].en.max()
        return (cond_a&cond_b)

def matched_file(pars,**kw):
    start, end, tot = kw['Coeffs']
    coefs = np.linspace(start, end, tot)

    # Create list of mean cluster energy from cluster_energy... file
    infile = common.fill_path(kw['EnergyIn'], **pars)
    #outfile = common.fill_path(kw['EnergyOut']+'_with_pt', **pars)
    outfile = common.fill_path(kw['EnergyOut'], **pars)

    with pd.HDFStore(infile,'r') as InFile, pd.HDFStore(outfile,'w') as OutFile:
    #with pd.HDFStore(infile,'r') as InFile:

        coef_keys = ['/coef_' + str(coef).replace('.','p') for coef in coefs]

        #for i, coef in enumerate(coef_keys[1:]):
        for coef in coef_keys:
            threshold = 0.05
            df_current = InFile[coef].reset_index(drop=True)

            df_current['deltar'] = deltar(df_current)

            df_current['matches'] = df_current.deltar<=threshold


            group=df_current.groupby('event')

            best_matches = group.apply(matching)
            if not isinstance(best_matches, pd.DataFrame):
                best_matches = best_matches.to_frame()

            df_current = df_current.set_index('event',append=True)

            df_current = df_current.join(best_matches, rsuffix='_max')

            df_current['event'] = df_current.index.get_level_values(1)
            df_current.index = df_current.index.get_level_values(0)

            df_current = df_current.rename(columns={0: "en_max"})

            if kw['BestMatch']:
                sel=df_current['en_max']==True
                df_current = df_current[sel]

            df_current = df_current.dropna()
            OutFile[coef] = df_current
            print("{} has been added to {}\n.".format(coef,outfile))

        print("{} has finished being filled.".format(outfile))

def effrms(df, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    out = {}
    x = np.sort(df, kind='mergesort')
    m = int(c *len(x)) + 1
    out = [np.min(x[m:] - x[:-m]) / 2.0]

    return out

def energy(pars, **kw):
    start, end, tot = kw['Coeffs']
    coefs = np.linspace(start, end, tot)

    # This will create a .hdf5 file containing the cluster energy information corresponding to each coefficient
    # By default this file is assumed to be there
    if kw['ReInit']:
        file = kw['File']
        clust_params = common.dict_per_file(params,file)['cluster']
        clust_params['ForEnergy'] = True
        for coef in coefs:
            clust_params['CoeffA'] = (coef,0)*50 # one for each layer
            print("Clustering with coef: ", coef)
            cluster_test.cluster(pars, **clust_params)

    if kw['MatchFile']:
        matched_file(pars, **kw)


if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing
    import matplotlib.pyplot as plt
    from matplotlib import cm

    parser = argparse.ArgumentParser(description='Clustering standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

    FLAGS.reg = 'All'
    FLAGS.sel = 'below_eta_2.7'

    input_files = params.fill_kw['FillInFiles']

    for key,files in input_files.items():
        if isinstance(files, str):
            files = [files]
        for file in files:
            energy_pars = common.dict_per_file(params,file)['energy']

            energy_pars['ReInit'] = True
            energy_pars['MatchFile'] = True

            energy(vars(FLAGS), **energy_pars)
