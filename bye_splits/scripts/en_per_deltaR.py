import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits import utils
from bye_splits import tasks
from bye_splits.utils import common
from bye_splits.tasks import cluster
from bye_splits.scripts import matching_v3 as match

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
    outfile = common.fill_path(input('Enter Output File Base Name: '), **pars)

    with pd.HDFStore(infile,'r') as InFile, pd.HDFStore(outfile,'w') as OutFile:

        coef_keys = ['/coef_' + str(coef).replace('.','p') for coef in coefs]
        #
        for coef in coef_keys[1:]:
            threshold = 0.05
            df_current = InFile[coef]

            df_current['deltar'] = deltar(df_current)

            df_current['matches'] = df_current.deltar<=threshold

            group=df_current.groupby('event')

            best_matches = group.apply(matching).to_frame()

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


def energy(pars, **kw):

    start, end, tot = kw['Coeffs']
    coefs = np.linspace(start, end, tot)
    normed_energies = np.array([])
    cell_nums = np.array([])

    # This will create a .hdf5 file containing the cluster energy information corresponding to each coefficient
    # By default this file is assumed to be there
    if kw['ReInit']:
        clust_params = params.cluster_kw
        clust_params['ForEnergy'] = True
        for coef in coefs:
            clust_params['CoeffA'] = (coef,0)*52 #28(EM)+24(FH+BH)
            cluster.cluster(pars, **clust_params)

    if kw['MatchFile']:
        matched_file(pars, **kw)

    file = common.fill_path(input('Input File Base Name: '), **pars)
    with pd.HDFStore(file,'r') as File:
        coefs = File.keys()
        max = File[coefs[-1]].set_index('event').drop(columns=['matches', 'en_max'])
        for i, coef in enumerate(coefs):

            if coef != coefs[-1]:
                df_current = File[coef].set_index('event').drop(columns=['matches', 'en_max'])

                df_joined = df_current.join(max,how='left',rsuffix='_max')
                df_joined = df_joined.T.drop_duplicates().T

                df_joined['normed_energies'] = df_joined['en']/df_joined['en_max']
                mean_energy = df_joined['normed_energies'].mean()
            else:
                mean_energy = 1.0

            normed_energies = np.append(normed_energies,mean_energy)

    one_line = np.full(tot-1,1.0)

    coef_ticks = coefs[0::5]

    coef_labels = [round(float(coef.split('_')[1].replace('p','.')),3) for coef in coefs]
    coef_labels= coef_labels[0::5]

    fig, ax = plt.subplots()
    ax.plot(coefs,normed_energies,color='blue')
    ax.plot(coefs,one_line,color='green')
    ax.set_xlabel(r'$R_{coef}$')
    ax.set_ylabel(r'$\frac{\bar{E}_{coef}}{\bar{E}_{max}}$')
    ax.set_title('Normalized Cluster Energy vs. Radius From Seed')
    ax.set_xticks(coef_ticks)
    ax.set_xticks(coefs, minor=True)
    ax.set_xticklabels(coef_labels)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.grid(which='major', alpha=0.5)
    plt.grid(which='minor', alpha=0.2)

    plt.savefig(input('Output Figure Name: ') + '.png', dpi=300)


if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Clustering standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

    energy(vars(FLAGS), **params.energy_kw)
