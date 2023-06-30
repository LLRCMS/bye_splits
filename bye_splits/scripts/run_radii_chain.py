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

def run_radii_chain(pars, particles, pu, coefs, event=None):
    df_gen, df_cl, df_tc = get_data_reco_chain_start(nevents=30, reprocess=False, particles=particles, pu=pu, event=event)

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
        filtered_keys = [key for key in store_file.keys() if key.startswith('/df_')]
        events = [int(key.split('df_')[1]) for key in filtered_keys]
        for key in filtered_keys:
            df = store_file.get(key)
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

if __name__ == "__main__":
    FLAGS = parsing.parser_radii_chain()
    if FLAGS.no_valid:
        FLAGS.no_valid_seed = True
        FLAGS.no_valid_cluster = True

    assert (FLAGS.sel in ("splits_only", "no_splits", "all") or
            FLAGS.sel.startswith("above_eta_"))

    run_radii_chain(common.dot_dict(vars(FLAGS)), particles=FLAGS.particles, PU=FLAGS.PU, coefs=FLAGS.coefs, event=FLAGS.event)
