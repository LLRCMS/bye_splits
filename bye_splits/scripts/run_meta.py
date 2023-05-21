# coding: utf-8

_all_ = [ ]

import os
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

import argparse

import tasks
from tasks import validation
import utils
from utils import params, common, parsing
from run_default_chain import run_default_chain
from run_roi_chain import run_roi_chain
from plot import chain_plotter

def avoid_single_chain_validation(pars):
    """
    Inhibits running the validation for a specific set of parameters.
    Used to control parameters of single chain scripts.
    """
    p = common.dot_dict(pars.copy())
    p.no_valid_seed = True
    p.no_valid_cluster = True
    return p

def run_meta(pars, user):
    single_pars = avoid_single_chain_validation(pars)

    print("======= Default Chain =========")
    run_default_chain(single_pars, user=user)

    print("======= ROI Chain =========")
    run_roi_chain(single_pars, user=user)

    base_d = params.read_task_params('base')

    # collect data from both chains
    collector = validation.Collector()
    # plot data from both chains together
    plotter = chain_plotter.ChainPlotter(chain_mode='both', user=user,
                                         tag='NEV'+str(pars.nevents))
    if not pars.no_valid_seed:
        out_df_seed = collector.collect_seed(pars, chain_mode='both', **base_d)
        plotter.seed_plotter(out_df_seed, pars)
        
    if not pars.no_valid_cluster:
        out_df_cluster = collector.collect_cluster(pars, chain_mode='both', **base_d)
        plotter.resolution_plotter(out_df_cluster, pars)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta script, running multiple reconstruction chains.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--no_fill",          action="store_true")
    parser.add_argument("--no_smooth",        action="store_true")
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

    run_meta(common.dot_dict(vars(FLAGS)), user="bfontana")
