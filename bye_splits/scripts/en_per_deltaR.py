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

            #breakpoint()

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

def plot_norm(pars, init_files, normby='pt'):
    #min_eta = 1.6
    plot_dict = {}
    normed_energies = dict.fromkeys(init_files.keys(),[0.0]) # Initialize at 0 since we only consider coefs[1:] (coefs[0] is an empty dataframe)
    rms_en_normed = dict.fromkeys(init_files.keys(),[0.0])
    eff_rms_en_normed = dict.fromkeys(init_files.keys(),[0.0])
    start = params.energy_kw['EnergyOut']
    for key in init_files.keys():
        #breakpoint()
        plot_dict[key] = [start+re.split('gen_cl3d_tc',file)[1] for file in init_files[key]]
        plot_dict[key] = [common.fill_path(file,**pars) for file in plot_dict[key]]
        if len(plot_dict[key])==1:
            with pd.HDFStore(plot_dict[key][0],'r') as File:
                coef_strs = File.keys()
                if normby=='max':
                    max = File[coef_strs[-1]].set_index('event').drop(columns=['matches','en_max'])
                for coef in coef_strs[1:]:
                    df = File[coef].set_index('event').drop(columns=['matches', 'en_max'])
                    if normby=='max':
                        df = df.join(max, on='event',rsuffix='_max')
                        df['normed_energies'] = df['en']/df['en_max']
                    elif normby=='pt':
                        #df['normed_energies'] = df['en']/df['genpart_energy']
                        df['normed_energies'] = df['cl3d_pt']/df['genpart_pt']
                    else:
                        df['normed_energies'] = df['en']/df['genpart_energy']

                    #df = df[ df['genpart_exeta'] > min_eta ]

                    mean_energy = df['normed_energies'].mean()

                    normed_energies[key] = np.append(normed_energies[key],mean_energy)
                    rms_en_normed[key] = np.append(rms_en_normed[key], df['normed_energies'].std()/mean_energy)
                    eff_rms_en_normed[key] = np.append(eff_rms_en_normed[key], effrms(df['normed_energies'])/mean_energy)
        else:
            file_list = [pd.HDFStore(val,'r') for val in plot_dict[key]]
            coef_strs = file_list[0].keys()
            if normby=='max':
                max = pd.concat([file_list[i][coef_strs[-1]].set_index('event').drop(columns=['matches','en_max']) for i in range(len(file_list))])
            for coef in coef_strs[1:]:
                df_list = [file_list[i][coef] for i in range(len(file_list))]

                full_df = pd.concat(df_list)
                full_df = full_df.set_index('event').drop(columns=['matches', 'en_max'])

                if normby=='max':
                    full_df = full_df.join(max,rsuffix='_max')
                    full_df['normed_energies'] = full_df['en']/full_df['en_max']
                elif normby=='pt':
                    full_df['normed_energies'] = full_df['cl3d_pt']/full_df['genpart_pt']
                else:
                    full_df['normed_energies'] = full_df['en']/full_df['genpart_energy']

                #full_df = full_df[ full_df['genpart_exeta'] > min_eta ]

                mean_energy = full_df['normed_energies'].mean()

                normed_energies[key] = np.append(normed_energies[key], mean_energy)
                rms_en_normed[key] = np.append(rms_en_normed[key], full_df['normed_energies'].std()/mean_energy)
                eff_rms_en_normed[key] = np.append(eff_rms_en_normed[key], effrms(full_df['normed_energies'])/mean_energy)

            for file in file_list:
                file.close()

    start, end, tot = params.energy_kw['Coeffs']
    coefs = np.linspace(start, end, tot)

    #one_line = np.full(tot-1,1.0)
    one_line = np.full(tot,1.0)

    coef_ticks = coefs[0::5]

    coef_labels = [round(coef,3) for coef in coefs]
    coef_labels= coef_labels[0::5]

    color_list = cm.rainbow(np.linspace(0,1,len(normed_energies)))

    for part,col in zip(normed_energies.keys(),color_list):
        en = normed_energies[part]
        rms = rms_en_normed[part]
        eff_rms = eff_rms_en_normed[part]

        fig, ax = plt.subplots(2,1, constrained_layout=True)
        fig.suptitle(part.replace('p','P'),fontsize=16)

        ax[0].plot(coefs,en,color=col)
        ax[0].plot(coefs,one_line,color='green',linestyle='dashed')

        ax[1].plot(coefs,rms,label='RMS', color=col)
        ax[1].plot(coefs,eff_rms,label=r'$RMS_{Eff}$',color=col,linestyle='dashed')

        ax[1].legend()

        ax[1].set_xlabel(r'$R_{coef}$')

        ax[0].set_title('Normalized Cluster Energy')
        ax[1].set_title('(Effective) RMS')

        ax[0].set_xticks(coef_ticks)
        ax[0].set_xticks(coefs, minor=True)
        ax[0].set_xticklabels([])

        ax[1].set_xticks(coef_ticks)
        ax[1].set_xticks(coefs, minor=True)
        ax[1].set_xticklabels(coef_labels)

        props = '\nalgo={}\nNormalization=genpart_{}'.format(pars['cluster_algo'],normby)

        if 'above' in pars['sel']:
            props = r'$\eta > 2.7$' + props
        elif 'below' in pars['sel']:
            props = r'$ 1.4 < \eta < 2.7$' + props

        ax[0].annotate(props, xy=(0.65,0.73), xycoords='figure fraction')

        plt.setp(ax[1].get_xticklabels(), rotation=90)

        plt.grid(which='major', alpha=0.5)
        plt.grid(which='minor', alpha=0.2)

        plt.savefig('{}_{}_{}_gen_{}_{}.png'.format(params.energy_kw['EnergyPlot'],pars['sel'],pars['cluster_algo'],normby,part), dpi=300)
        plt.close()


def energy(pars, **kw):
    start, end, tot = kw['Coeffs']
    coefs = np.linspace(start, end, tot)

    # This will create a .hdf5 file containing the cluster energy information corresponding to each coefficient
    # By default this file is assumed to be there
    if kw['ReInit']:
        #clust_params = params.cluster_kw
        file = kw['File']
        clust_params = common.dict_per_file(params,file)['cluster']
        clust_params['ForEnergy'] = True
        for coef in coefs:
            clust_params['CoeffA'] = (coef,0)*52 #28(EM)+24(FH+BH)
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

            energy_pars['ReInit'] = False
            energy_pars['MatchFile'] = False

            energy(vars(FLAGS), **energy_pars)

    if params.energy_kw['MakePlot']:
        plot_norm(vars(FLAGS),input_files,normby='en')
