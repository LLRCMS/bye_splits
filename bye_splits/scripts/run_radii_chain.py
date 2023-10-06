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

with open(params.CfgPath, "r") as afile:
    cfg = yaml.safe_load(afile)

def get_cluster_data(cluster_params, chain, coefs, pars):
    cluster_data = {}
    
    for coef in coefs:
        cluster_params["CoeffA"] = [coef, 0] * 50
        if chain == 'default': 
            nevents = tasks.cluster.cluster_default(pars, **cluster_params)
            store_file = pd.HDFStore(common.fill_path(cluster_params["ClusterOutPlot"], **pars), mode='r')
        else:
            nevents = tasks.cluster.cluster_cs(pars, **cluster_params)
            store_file = pd.HDFStore(common.fill_path(cluster_params["ClusterOutPlotCS"], **pars), mode='r')
        
        # Extract coefs and events
        df_list = []
        filtered_keys = [key for key in store_file.keys() if key.startswith('/df_')]
        events = [int(key.split('df_')[1]) for key in filtered_keys]

        for key in filtered_keys:
            df = store_file.get(key)
            df_list.append(df)

        cluster_data[str(coef)[2:]] = df_list
        store_file.close()
    
    return cluster_data, events

def merge_cluster_data_with_event(df_event_tc, cluster_data, index):
    """ data processed from clustering step is merged with original data events
        to get infomation about the non-clusterised TCs """
    merged_data = {}
    
    for coef in cluster_data.keys():
        merged_data[coef] = pd.merge(
            left=cluster_data[coef][index],
            right=df_event_tc[~df_event_tc['tc_layer'].isin(cfg['selection']['disconnectedTriggerLayers'])],
            on=['tc_wu', 'tc_wv', 'tc_cu', 'tc_cv', 'tc_layer'],
            how='outer'
        ).fillna(cluster_data[coef][index]['seed_idx'].max() + 1)
    
    return merged_data

def run_radii_chain(pars, particles, pu, coefs, event=None):
    df_gen, df_cl, df_tc = get_data_reco_chain_start(nevents=30, reprocess=False, particles=particles, pu=pu, event=event)

    # Default chain
    fill_params = params.read_task_params("fill")
    tasks.fill.fill(pars, df_gen, df_cl, df_tc, **fill_params)

    smooth_params = params.read_task_params("smooth")
    tasks.smooth.smooth(pars, **smooth_params)

    seed_params = params.read_task_params("seed")
    tasks.seed.seed(pars, **seed_params)

    cluster_params = params.read_task_params("cluster")
    cluster_data_default, _ = get_cluster_data(cluster_params, 'default', coefs, pars)

    # Coarse seeding chain
    fill_cs_params = params.read_task_params('cs')
    tasks.coarse_seeding.coarse_seeding(pars, df_gen, df_cl, df_tc, **fill_cs_params)

    seed_cs_params = params.read_task_params('seed_cs')
    tasks.seed_cs.seed_cs(pars, **seed_cs_params)

    cluster_params = params.read_task_params("cluster")
    cluster_data_cs, events = get_cluster_data(cluster_params, 'cs', coefs, pars)

    # Data unpacker
    dict_event = {}
    
    for index, ev in enumerate(events):
        dict_event[ev] = {}
        df_event_tc = df_tc[df_tc.event == ev][['tc_mipPt', 'tc_eta', 'tc_wu', 'tc_wv', 'tc_cu', 'tc_cv', 'tc_layer']]
        
        # Merge cluster data with event data
        dict_event[ev]['default'] = merge_cluster_data_with_event(df_event_tc, cluster_data_default, index)
        dict_event[ev]['cs'] = merge_cluster_data_with_event(df_event_tc, cluster_data_cs, index)

    return dict_event, df_gen


if __name__ == "__main__":
    FLAGS = parsing.parser_radii_chain()
    if FLAGS.no_valid:
        FLAGS.no_valid_seed = True
        FLAGS.no_valid_cluster = True

    assert (FLAGS.sel in ("splits_only", "no_splits", "all") or
            FLAGS.sel.startswith("above_eta_"))

    run_radii_chain(common.dot_dict(vars(FLAGS)), particles=FLAGS.particles, PU=FLAGS.PU, coefs=FLAGS.coefs, event=FLAGS.event)
