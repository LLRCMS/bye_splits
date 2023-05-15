# coding: utf-8

_all_ = [ "run_default_chain" ]

import os
from pathlib import Path
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

import yaml
import tasks
import utils
from utils import params, common, parsing
import data_handle
from data_handle.data_process import EventDataParticle, get_data_reco_chain_start
from data_handle.geometry import GeometryData
import plot
from plot import chain_plotter
from tasks import validation

import argparse
import pandas as pd

def run_chain_radii(pars, particles, coefs, event=None):
    df_gen, df_cl, df_tc = get_data_reco_chain_start(nevents=300, particles=particles, event=event)

    fill_d = params.read_task_params("fill")
    tasks.fill.fill(pars, df_gen, df_cl, df_tc, **fill_d)

    smooth_d = params.read_task_params("smooth")
    tasks.smooth.smooth(pars, **smooth_d)

    seed_d = params.read_task_params("seed")
    tasks.seed.seed(pars, **seed_d)

    cluster_d = params.read_task_params("cluster")
    dict_cluster = {}
    for coef in coefs:
        cluster_d["CoeffA"] = [coef, 0] * 50
        nevents = tasks.cluster.cluster_default(pars, **cluster_d)
        store_file = pd.HDFStore(common.fill_path(cluster_d["ClusterOutPlot"], **pars), mode='r')
        df_list = []
        filtered_keys = [key for key in store_file.keys() if key.startswith('df_')]
        events = [key.split('df_')[1] for key in filtered_keys]
        for key in filtered_keys:
            df = store.get(key)
            df_list.append(df)
        dict_cluster[str(coef)[2:]] = df_list
        store_file.close() 

    dict_event = {}
    for index, ev in enumerate(events):
        dict_event[ev] = {}
        df_event_tc = df_tc[df_tc.event == ev][['tc_mipPt','tc_eta','tc_wu','tc_wv','tc_cu','tc_cv','tc_layer']]
        for coef in dict_cluster.keys():
            dict_event[ev][coef] = pd.merge(left=dict_cluster[coef][index], 
                                            right=df_event_tc[df_event_tc.tc_layer%2 != 0], 
                                            on=['tc_wu', 'tc_wv', 'tc_cu', 'tc_cv', 'tc_layer'],
                                            how='outer').fillna(dict_cluster[coef][index]['seed_idx'].max()+1) 
    return dict_event, df_gen

def run_default_chain(pars, user):
    """Run the backend stage 2 reconstruction chain for a single event."""
    df_out = None
    collector = validation.Collector()
    plotter = chain_plotter.ChainPlotter(chain_mode='default', user=user,
                                         tag='NEV'+str(pars.nevents))
    df_gen, df_cl, df_tc = get_data_reco_chain_start(nevents=pars.nevents,
                                                     reprocess=True, tag="default_chain")

    print("There are {} events in the input.".format(df_gen.shape[0]))

    if not pars.no_fill:
        fill_d = params.read_task_params("fill")
        tasks.fill.fill(pars, df_gen, df_cl, df_tc, **fill_d)

    if not pars.no_smooth:
        smooth_d = params.read_task_params("smooth")
        tasks.smooth.smooth(pars, **smooth_d)

    if not pars.no_seed:
        seed_d = params.read_task_params("seed")
        tasks.seed.seed(pars, **seed_d)

    if not pars.no_valid_seed:
        valid_d = params.read_task_params('valid_seed_default')
        stats_out_seed = collector.collect_seed(pars, chain_mode='default', **valid_d)
        plotter.seed_plotter(stats_out_seed, pars)

    nparameters = 1
    for _ in range(nparameters):  # clustering optimization studies
        if not pars.no_cluster:
            cluster_d = params.read_task_params("cluster")
            nevents_end = tasks.cluster.cluster_default(pars, **cluster_d)
            print("There are {} events in the output.".format(nevents_end))

        if not pars.no_valid_cluster:
            valid_d = params.read_task_params("valid")
            # validation.validation_cmssw(pars, **valid_d) # compare CMSSW with local reconstruction

            stats_out = collector.collect_cluster(pars, chain_mode='default', **valid_d)
            if df_out is not None:
                df_out = stats_out
            else:
                df_out = pd.concat((df_out, stats_out), axis=0)

    if not pars.no_valid_cluster:
        plotter.resolution_plotter(df_out, pars)
        #plotter.distribution_plotter(df_out, pars)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full reconstruction chain.")
    parser.add_argument("--no_fill",          action="store_true")
    parser.add_argument("--no_smooth",        action="store_true")
    parser.add_argument("--no_seed",          action="store_true")
    parser.add_argument("--no_cluster",       action="store_true")
    parser.add_argument('--no_valid',         action='store_true', help='do not run any validation')
    parser.add_argument('--no_valid_seed',    action='store_true', help='do not run ROI seed validation')
    parser.add_argument('--no_valid_cluster', action='store_true', help='do not run ROI cluster validation')

    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    if FLAGS.no_valid:
        FLAGS.no_valid_seed = True
        FLAGS.no_valid_cluster = True

    assert (FLAGS.sel in ("splits_only", "no_splits", "all") or
            FLAGS.sel.startswith("above_eta_"))

    run_default_chain(common.dot_dict(vars(FLAGS)), user='bfontana')
