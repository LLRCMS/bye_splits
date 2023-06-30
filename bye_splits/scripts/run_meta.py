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
from run_new_chain import run_new_chain
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

    print("======= New Chain =========")
    run_new_chain(single_pars, user=user)

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
    FLAGS = parsing.parser_meta()
    if FLAGS.no_valid:
        FLAGS.no_valid_seed = True
        FLAGS.no_valid_cluster = True
        
    assert (FLAGS.sel in ('splits_only', 'no_splits', 'all') or
            FLAGS.sel.startswith('above_eta_'))

    run_meta(common.dot_dict(vars(FLAGS)), user=FLAGS.user)
