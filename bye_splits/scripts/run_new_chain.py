# coding: utf-8

_all_ = [ "run_new_chain" ]

import os
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

import tasks
import utils
from utils import params, common, parsing
import data_handle
from data_handle.data_process import get_data_reco_chain_start
import plot
from plot import chain_plotter
from tasks import validation

import argparse

def run_new_chain(pars, user='bfontana'):
    '''Run the backend stage 2 reconstruction chain for a single event.'''
    collector = validation.Collector()
    plotter = chain_plotter.ChainPlotter(chain_mode='cs', user=user,
                                         tag='NEV'+str(pars.nevents))
    df_gen, df_cl, df_tc = get_data_reco_chain_start(nevents=pars.nevents,
                                                     reprocess=True, tag='new_chain')

    print('There are {} events in the input.'.format(df_gen.shape[0]))

    if not pars.no_cs:
        fill_d = params.read_task_params('cs')
        tasks.coarse_seeding.coarse_seeding(pars, df_gen, df_cl, df_tc, **fill_d)

    if not pars.no_seed:
        seed_d = params.read_task_params('seed_cs')
        tasks.seed_cs.seed_cs(pars, **seed_d)

    if not pars.no_valid_seed:
        valid_d = params.read_task_params('valid_seed_cs')
        stats_out_seed = collector.collect_seed(pars, chain_mode='cs', **valid_d)
        plotter.seed_plotter(stats_out_seed, pars)

    if not pars.no_cluster:
        cluster_d = params.read_task_params('cluster')
        nevents_end = tasks.cluster.cluster_cs(pars, **cluster_d)
        print('There are {} events in the output.'.format(nevents_end))

    if not pars.no_valid_cluster:
        # compare CMSSW with local reconstruction
        valid_d = params.read_task_params('valid_cluster')

        # validate the clustering
        stats_out_cluster = collector.collect_cluster(pars, chain_mode='cs', **valid_d)
        plotter.resolution_plotter(stats_out_cluster, pars)
        #plotter.distribution_plotter(stats_out_cluster, pars)

    return None


if __name__ == '__main__':
    FLAGS = parsing.parser_new_chain()
    if FLAGS.no_valid:
        FLAGS.no_valid_seed = True
        FLAGS.no_valid_cluster = True
        
    assert (FLAGS.sel in ('splits_only', 'no_splits', 'all') or
            FLAGS.sel.startswith('above_eta_'))

    run_new_chain(common.dot_dict(vars(FLAGS)), user=FLAGS.user)
