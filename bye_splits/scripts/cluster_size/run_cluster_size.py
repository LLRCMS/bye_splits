# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing, cl_helpers

from bye_splits.scripts.iterative_optimization import optimization
from data_handle.data_handle import get_data_reco_chain_start

import csv
import argparse
import random
import re

random.seed(10)
import numpy as np
import pandas as pd
import sys

import yaml
import uproot as up

from tqdm import tqdm


def normalize_df(cl_df, gen_df):
    cl_df["pt"] = cl_df["en"] / np.cosh(cl_df["eta"])
    gen_df["gen_pt"] = gen_df["gen_en"] / np.cosh(gen_df["gen_eta"])

    cl_df = cl_df.set_index("event").join(
        gen_df.set_index("event"), on="event", how="inner"
    )

    cl_df["pt_norm"] = cl_df["pt"] / cl_df["gen_pt"]
    cl_df["en_norm"] = cl_df["en"] / cl_df["gen_en"]

    return cl_df


def combine_files_by_coef(in_dir, out_path):
    file_pattern = os.path.basename(out_path).replace(".hdf5", "")
    files = [
        file for file in os.listdir(in_dir) if re.search(file_pattern, file) != None
    ]

    coef_pattern = r"coef_0p(\d+)"
    with pd.HDFStore(out_path, "w") as clusterSizeOut:
        for file in files:
            key = re.search(coef_pattern, file).group()
            with pd.HDFStore(in_dir + "/" + file, "r") as clSizeCoef:
                clusterSizeOut[key] = clSizeCoef["/data"]


def cluster_size(pars, cfg):
    df_out = None
    nevents = cfg["clusterStudies"]["nevents"]

    cluster_d = params.read_task_params("cluster")

    if cfg["clusterStudies"]["reinit"]:
        df_gen, df_cl, df_tc = get_data_reco_chain_start(
            nevents=nevents, reprocess=True
        )

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

    if not pars.no_cluster:
        start, end, tot = cfg["clusterStudies"]["coeffs"]

        coefs = np.linspace(start, end, tot)
        print("\nIterating over cluster radii.\n")
        for coef in tqdm(coefs, total=len(coefs)):
            cl_size_coef = "{}_coef_{}".format(
                cfg["clusterStudies"]["clusterSizeBaseName"],
                str(round(coef, 3)).replace(".", "p"),
            )
            cluster_d["ClusterOutPlot"] = cl_size_coef
            cluster_d["CoeffA"] = [coef, 0] * 50
            nevents_end = tasks.cluster.cluster(pars, **cluster_d)

        cl_size_out = common.fill_path(cfg["clusterStudies"]["clusterSizeBaseName"])

        combine_files_by_coef(params.LocalStorage, cl_size_out)

        with pd.HDFStore(cl_size_out, mode="a") as clSizeOut:
            df_gen, _, _ = get_data_reco_chain_start(
                nevents=nevents, reprocess=False, tag="cluster_size"
            )
            coef_keys = clSizeOut.keys()
            for coef in coef_keys[1:]:
                clSizeOut[coef] = normalize_df(clSizeOut[coef], df_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-r",
        "--process",
        help="reprocess trigger cell geometry data",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--plot",
        help="plot shifted trigger cells instead of originals",
        action="store_true",
    )
    parser.add_argument("--no_fill", action="store_true")
    parser.add_argument("--no_smooth", action="store_true")
    parser.add_argument("--no_seed", action="store_true")
    parser.add_argument("--no_cluster", action="store_true")
    #nevents_help = "Number of events for processing. Pass '-1' for all events."
    #parser.add_argument("-n", "--nevents", help=nevents_help, default=-1, type=int)
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    cluster_size(common.dot_dict(vars(FLAGS)), cfg)
