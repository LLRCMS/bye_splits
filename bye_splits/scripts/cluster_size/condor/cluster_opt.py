# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing, cl_helpers

from data_handle.data_process import get_data_reco_chain_start

import argparse
import random

random.seed(10)
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

import yaml

pt_norm_loss = lambda mean: np.sqrt(1-mean**2)/2

def get_gen(pars, cfg):    
    particles = pars["particles"]
    eta = pars["eta"]

    reprocess = cfg["clusterStudies"]["reprocess"]
    nevents = cfg["clusterStudies"]["nevents"]
    tag = cfg["clusterStudies"]["parquetTag"]

    df_gen, _ , _= get_data_reco_chain_start(
        particles=particles, nevents=nevents, reprocess=reprocess, tag=tag
    )

    df = df_gen[ df_gen.gen_eta > 0 ] if eta=="pos" else df_gen[ df_gen.gen_eta < 0 ]
    if eta=="neg": df_gen["gen_eta"] = abs(df_gen.gen_eta)

    return df

def cluster_coef(pars, cfg, radii):
    particles = pars["particles"]
    cluster_d = params.read_task_params("cluster")

    cluster_d["ClusterOutPlot"], cluster_d["ClusterOutValidation"] = cfg["clusterStudies"]["clusterSizeBaseName"], cfg["clusterStudies"]["clusterSizeBaseName"]+"_valid"
    cluster_d["CoeffA"] = radii

    for key in ("ClusterInTC", "ClusterInSeeds", "ClusterOutPlot", "ClusterOutValidation"):
        name = cluster_d[key]

        cluster_d[key] =  "{}_PU0_{}_posEta".format(particles, name)
    
    cluster_d["returnDF"] = True
    _, df = tasks.cluster.cluster_default(pars, **cluster_d)
    
    return df

def normalize_df(cl_df, gen_df, dRThresh=0.05):
    cl_df=cl_df.reset_index().set_index(["event","seed_idx"])
    combined_df = cl_df.join(
        gen_df.set_index("event"), on="event", how="inner"
    )

    if "dR" not in combined_df.keys():
        combined_df["dR"] = np.sqrt((abs(combined_df["eta"])-abs(combined_df["gen_eta"]))**2+(combined_df["phi"]-combined_df["gen_phi"])**2)
    if "matches" not in combined_df.keys():
        combined_df["matches"] = combined_df["dR"] <= dRThresh

    combined_df["pt"] = combined_df["en"] / np.cosh(combined_df["eta"])
    combined_df["gen_pt"] = combined_df["gen_en"] / np.cosh(combined_df["gen_eta"])

    combined_df["pt_norm"] = combined_df["pt"] / combined_df["gen_pt"]
    combined_df["en_norm"] = combined_df["en"] / combined_df["gen_en"]

    return combined_df

def filter_df(df):
    df = df[ df.matches == True ]
    df = df.groupby("event").apply(lambda x: x.loc[x.pt.idxmax()])
    
    return df

class clusterRad:
    def __init__(self, pars, cfg):
        self.pars = pars
        self.cfg = cfg
        self.df_gen = get_gen(pars, cfg)

    def cluster_check(self, radii):
        df_cluster = cluster_coef(self.pars, self.cfg, radii)
        df = normalize_df(df_cluster, self.df_gen)
        return filter_df(df)

    def cluster_loss(self, radii):
        df_cluster = cluster_coef(self.pars, self.cfg, radii)
        df = normalize_df(df_cluster, self.df_gen)
        df_filt = filter_df(df)
        pt_norm_mean = df_filt.pt_norm.mean()
        
        return pt_norm_loss(pt_norm_mean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--particles", choices=("photons", "electrons", "pions"), required=True)
    parser.add_argument("--eta", help="Eta region", choices=("pos", "neg"), default="pos")
    parsing.add_parameters(parser)
    
    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)
    
    if pars.particles == "photons":
        #init_radii = np.asarray([0.01]*50)
        init_radii = np.asarray([0.042, 0.042, 0.014, 0.014, 0.017, 0.17, 0.011, 0.011, 0.014, 0.014, 0.011, 0.011, 0.013, 0.013, 0.012, 0.012, 0.026, 0.026, 0.021, 0.021, 0.021, 0.021, 0.016, 0.016, 0.029, 0.029, 0.042, 0.042])
    else:
        init_radii = np.asarray([0.015]*50) if pars.particles == "electrons" else np.asarray([0.02]*50)

    cluster = clusterRad(pars, cfg)
    test_df = cluster.cluster_check(init_radii)
    breakpoint()


    '''lower_bounds, upper_bounds = 0.5*init_radii, 1.5*init_radii
    bounds = Bounds(lower_bounds, upper_bounds)    


    min_options = {"maxiter": 2}

    res = minimize(cluster.cluster_loss, init_radii, bounds=bounds, options=min_options)'''



