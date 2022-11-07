#!/usr/bin/env python

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import tqdm
import uproot # uproot4
from datetime import date
import optparse
from itertools import chain
import functools
import operator
from bye_splits.utils import common

disconnectedTriggerLayers = [
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28
]

def deltar(df):
    df['deta']=df['cl3d_eta']-df['genpart_exeta']
    df['dphi']=np.abs(df['cl3d_phi']-df['genpart_exphi'])
    sel=df['dphi']>np.pi
    df['dphi']-=sel*(2*np.pi)
    return(np.sqrt(df['dphi']*df['dphi']+df['deta']*df['deta']))

def matching(event):
    if event.matches.sum()==0:
        return event.cl3d_pt==event.cl3d_pt.max()
    else:
        cond_a = event.matches==True
        cond_b = event.cl3d_pt==event[cond_a].cl3d_pt.max()
        return (cond_a&cond_b)

def transform(nested_list):
    regular_list=[]
    for ele in nested_list:
        if type(ele) is list:
            regular_list.append(ele)
        else:
            regular_list.append([ele])
    return regular_list

def create_dataframes(pars, **kw):
    print('Input file: {}'.format(kw['File']), flush=True)

    branches_gen = [ 'event', 'genpart_reachedEE', 'genpart_pid', 'genpart_gen',
                     'genpart_exphi', 'genpart_exeta', 'genpart_energy' ]

    branches_cl3d = [ 'event', 'cl3d_energy','cl3d_pt','cl3d_eta','cl3d_phi' ]
    branches_tc = [ 'event', 'tc_zside', 'tc_energy', 'tc_mipPt', 'tc_pt', 'tc_layer',
                    'tc_x', 'tc_y', 'tc_z', 'tc_phi', 'tc_eta', 'tc_id' ]

    batches_gen, batches_tc = ([] for _ in range(2))
    memsize_gen, memsize_tc = '128 MB', '64 MB'

    #with uproot.open(filename + ':' + kw['GenTree'][key]) as data:
    with uproot.open(kw['File'] + ':' + kw['GenTree']) as data:
        #breakpoint()
        #print( data.num_entries_for(memsize, expressions=branches_tc) )
        for ib,batch in enumerate(data.iterate(branches_gen, step_size=memsize_gen,
                                               library='pd')):
            # reachedEE=2: photons that hit HGCAL
            batch = batch[ batch['genpart_reachedEE']==kw['ReachedEE'] ]
            batch = batch[ batch['genpart_gen']!=-1 ]
            #batch = batch[ batch['genpart_pid']==22 ] # Set this to be the pid of each incoming particle
            #batch = batch.drop(columns=['genpart_reachedEE', 'genpart_gen', 'genpart_pid'])
            batch = batch.drop(columns=['genpart_gen', 'genpart_pid'])
            batch = batch[ batch['genpart_exeta']>0  ] #positive endcap only
            batch.set_index('event', inplace=True)

            batches_gen.append(batch)
            print('Step {}: +{} generated data processed.'.format(ib,memsize_gen), flush=True)

        for ib,batch in enumerate(data.iterate(branches_tc, step_size=memsize_tc,
                                               library='pd')):

            batch = batch[ batch['tc_zside']==1 ] #positive endcap
            batch = batch.drop(columns=['tc_zside'])
            #remove layers not read by trigger cells
            batch = batch[ ~batch['tc_layer'].isin(disconnectedTriggerLayers) ]
            #convert all the trigger cell hits in each event to a list
            batch = batch.groupby(by=['event']).aggregate(lambda x: list(x))
            batches_tc.append(batch)
            print('Step {}: +{} trigger cells data processed.'.format(ib,memsize_tc), flush=True)

    df_gen = pd.concat(batches_gen)
    df_tc = pd.concat(batches_tc)

    df_algos = {}
    #assert len(files)==1 #modify the following block otherwise # IMPORTANT
    for algo_name, algo_tree in kw['AlgoTree'].items():
        with uproot.open(kw['File'])[algo_tree] as tree:
            df_algos[algo_name] = tree.arrays(branches_cl3d + ['cl3d_layer_pt'], library='pd')
            df_algos[algo_name].reset_index(inplace=True)

            # Trick to expand layers pTs, which is a vector of vector
            newcol = df_algos[algo_name].apply(lambda row: row.cl3d_layer_pt[row.subentry], axis=1)
            df_algos[algo_name]['cl3d_layer_pt'] = newcol
            df_algos[algo_name] = df_algos[algo_name].drop(['subentry', 'entry'], axis=1)


    return (df_gen, df_algos, df_tc)

def preprocessing(pars, **kw):

    gen, algo, tc = create_dataframes(pars, **kw)

    algo_clean={}

    # split df_gen_clean in two, one collection for each endcap
    #gen_neg = gen_clean[ gen_clean['genpart_exeta']<=0 ]
    #gen_pos = gen[ gen['genpart_exeta']>0  ]
    #df_gen = df_gen.join(pd.concat(batches_tc), how='left', rsuffix='_tc')

    for algo_name,df_algo in algo.items():
        # split clusters in two, one collection for each endcap
        algo_pos = df_algo[ df_algo['cl3d_eta']>0  ]

        #algo_neg = df_algo[ df_algo['cl3d_eta']<=0 ]

        #set the indices
        algo_pos.set_index('event', inplace=True)
        #algo_neg.set_index('event', inplace=True)

        #merging gen columns and cluster columns, keeping cluster duplicates (same event)
        algo_pos_merged=gen.join(algo_pos, how='right', rsuffix='_algo').dropna()

        #algo_neg_merged=gen_neg.join(algo_neg, how='left', rsuffix='_algo')

        # compute deltar
        algo_pos_merged['deltar']=deltar(algo_pos_merged)
        #algo_neg_merged['deltar']=deltar(algo_neg_merged)

        #could be better:
        algo_pos_merged['matches'] = algo_pos_merged.deltar<=kw['Threshold']
        #algo_neg_merged['matches'] = algo_neg_merged.deltar<=threshold

        #matching
        # /!\ LP: but then, we want to remove only clusters that aren't "best match"
        #         best match could be:
        #              - Unmatched cluster with highest pT if no dr-matched cluster in evt
        #              - Matched cluster with highest pT *among dr-matched clusters*
        group=algo_pos_merged.groupby('event') # required when dealing with pile-up

        best_matches = group.apply(matching)
        if isinstance(best_matches, pd.DataFrame):
            algo_pos_merged['best_match']=best_matches.array
        else:
            best_matches = best_matches.to_frame()
            #algo_pos_merged = algo_pos_merged.set_index('event',append=True)
            best_matches.index = best_matches.index.droplevel(1)
            algo_pos_merged = algo_pos_merged.join(best_matches)
            algo_pos_merged = algo_pos_merged.rename(columns={0: 'best_match'})

        #algo_pos_merged['best_match']=group.apply(matching).array
        #group=algo_neg_merged.groupby('event')
        #algo_neg_merged['best_match']=group.apply(matching).array

        #keep matched clusters only
        if kw['BestMatch']:
            sel=algo_pos_merged['best_match']==True
            algo_pos_merged=algo_pos_merged[sel]

            #sel=algo_neg_merged['best_match']==True
            #algo_neg_merged=algo_neg_merged[sel]

        #algo_clean[algo_name]=pd.concat([algo_neg_merged,algo_pos_merged], sort=False).sort_values('event')
        algo_clean[algo_name] = algo_pos_merged.sort_values('event')
        algo_clean[algo_name] = algo_clean[algo_name].join(tc, how='left', rsuffix='_tc')


    #save files to savedir in HDF
    outfile = kw['OutFile']
    store = pd.HDFStore(outfile, mode='w')
    for algo_name, df in algo_clean.items():
        store[algo_name] = df
        store.close()

def match(pars, **kw):
    files = kw['Files']
    for key, val in files.items():
        file_list  = val
        if isinstance(file_list, str):
            file_list = [file_list]
        for file in file_list:
            for tree in kw['AlgoTrees'].keys():
                kw['File'] = file
                outfile = 'data/gen_cl3d_tc_{}_{}.hdf5'.format(tree, re.split('.root|/',file)[-2])
                kw['OutFile'] = outfile
                kw['AlgoTree'] = {tree: kw['AlgoTrees'][tree][key]}
                kw['GenTree'] = kw['GenTrees'][key]
                #print('Reading data from {} and creating file {}.'.format(file,outfile))
                preprocessing(pars, **kw)

#Run with: `python scripts/matching_v3.py
if __name__=='__main__':
    import argparse
    import itertools
    import re
    from glob import glob
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description='Matching standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

    match(vars(FLAGS), **params.match_kw)
