# coding: utf-8

_all_ = [ ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing, cl_helpers

from bye_splits.iterative_optimization import optimization
import data_handle
from data_handle.data_handle import get_data_reco_chain_start

import csv
import argparse
import random; random.seed(10)
import numpy as np
import pandas as pd
import sys

import yaml
import uproot as up

def normalize_df(cl_df, gen_df):
    cl_df['pt'] = cl_df['en']/np.cosh(cl_df['eta'])
    gen_df['gen_pt'] = gen_df['gen_en']/np.cosh(gen_df['gen_eta'])
    
    cl_df = cl_df.set_index('event').join(gen_df.set_index('event'),on='event', how='inner')

    cl_df['pt_norm'] = cl_df['pt']/cl_df['gen_pt']
    cl_df['en_norm'] = cl_df['en']/cl_df['gen_en']

    return cl_df

def cluster_size(pars, cfg):
    df_out = None

    cluster_d = params.read_task_params('cluster')

    if cfgprod['cl_size']['reinit']:
        df_gen, df_cl, df_tc = get_data_reco_chain_start('cluster_size', nevents=10, reprocess=True)

        print('There are {} events in the input.'.format(df_gen.shape[0]))

        if not pars.no_fill:
            fill_d = params.read_task_params('fill')
            tasks.fill.fill(pars, df_gen, df_cl, df_tc, **fill_d)
            
        if not pars.no_smooth:
            smooth_d = params.read_task_params('smooth')
            tasks.smooth.smooth(pars, **smooth_d)
                
        if not pars.no_seed:
            seed_d = params.read_task_params('seed')
            tasks.seed.seed(pars, **seed_d)

    if not pars.no_cluster:
            cluster_d['GenerateClusterSize'] = True
            
            cl_size_out = common.fill_path(cfgprod['cl_size']['ClusterSizeBaseName'], **pars)
            cl_size_out = cl_helpers.update_version_name(cl_size_out)
            cluster_d['ClusterSizePath'] = cl_size_out

            start, end, tot = cfgprod['cl_size']['Coeffs']
            coefs = np.linspace(start, end, tot)
            for coef in coefs:
                cluster_d['CoeffA'] = [coef, 0]*50
                print(f"\nStarting Coefficient: {coef}\n")
                nevents_end = tasks.cluster.cluster(pars, **cluster_d)

            with pd.HDFStore(cl_size_out, mode='a') as clSizeOut:
                df_gen, _, _ = get_data_reco_chain_start('cluster_size', nevents=10, reprocess=False)
                coef_keys = clSizeOut.keys()
                for coef in coef_keys[1:]:
                    clSizeOut[coef] = normalize_df(clSizeOut[coef], df_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--process',
                        help='reprocess trigger cell geometry data',
                        action='store_true')
    parser.add_argument('-p', '--plot',
                        help='plot shifted trigger cells instead of originals',
                        action='store_true')
    parser.add_argument('--no_fill',    action='store_true')
    parser.add_argument('--no_smooth',  action='store_true')
    parser.add_argument('--no_seed',    action='store_true')
    parser.add_argument('--no_cluster', action='store_true')
    nevents_help = "Number of events for processing. Pass '-1' for all events."
    parser.add_argument('-n', '--nevents', help=nevents_help,
                        default=-1, type=int)
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()

    with open(params.CfgPaths['cluster_size'], 'r') as afile:
        cfgprod = yaml.safe_load(afile)

    cluster_size(common.dot_dict(vars(FLAGS)), cfgprod)