# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

import tasks
import utils
from utils import params, common, parsing
import data_handle
from data_handle.data_process import EventDataParticle, get_data_reco_chain_start
from data_handle.geometry import GeometryData
import plot
from plot import roi_chain_plotter

import argparse

def run_roi_chain(pars):
    '''Run the backend stage 2 reconstruction chain for a single event.'''
    df_gen, df_cl, df_tc = get_data_reco_chain_start(nevents=100, reprocess=False, tag='roi_chain')

    print('There are {} events in the input.'.format(df_gen.shape[0]))

    if not pars.no_roi:
        fill_d = params.read_task_params('roi')
        tasks.roi.roi(pars, df_gen, df_cl, df_tc, **fill_d)

    if not pars.no_seed:
        seed_d = params.read_task_params('seed_roi')
        tasks.seed_roi.seed_roi(pars, **seed_d)

    if not pars.no_valid_seed:
        valid_d = params.read_task_params('valid_seed')
        stats_out_seed = tasks.validation_roi.stats_collector_seed(pars, **valid_d)
        roi_chain_plotter.seed_plotter(stats_out_seed, pars, user='bfontana')

    if not pars.no_cluster:
        cluster_d = params.read_task_params('cluster')
        nevents_end = tasks.cluster.cluster_roi(pars, **cluster_d)
        print('There are {} events in the output.'.format(nevents_end))

    if not pars.no_valid_cluster:
        # compare CMSSW with local reconstruction
        valid_d = params.read_task_params('valid_cluster')

        # validate the clustering
        stats_out_cluster = tasks.validation_roi.stats_collector_roi(pars, mode='resolution', **valid_d)
        roi_chain_plotter.resolution_plotter(stats_out_cluster, pars, user='bfontana')
        roi_chain_plotter.distribution_plotter(stats_out_cluster, pars, user='bfontana')

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Full reconstruction chain.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--no_roi',           action='store_true', help='do not run the roi finder step')
    parser.add_argument('--no_seed',          action='store_true', help='do not run the seeding step')
    parser.add_argument('--no_cluster',       action='store_true', help='do not run the clustering step')
    parser.add_argument('--no_valid',         action='store_true', help='do not run any validation')
    parser.add_argument('--no_valid_seed',    action='store_true', help='do not run ROI seed validation')
    parser.add_argument('--no_valid_cluster', action='store_true', help='do not run ROI cluster validation')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    if FLAGS.no_valid:
        FLAGS.no_valid_seed = True
        FLAGS.no_valid_cluster = True
        
    assert (FLAGS.sel in ('splits_only', 'no_splits', 'all') or
            FLAGS.sel.startswith('above_eta_'))

    run_roi_chain(common.dot_dict(vars(FLAGS)))
