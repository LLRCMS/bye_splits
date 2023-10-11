# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing, cl_helpers

import argparse
import random

random.seed(10)
import numpy as np
import pandas as pd

import yaml

def cluster_radius(pars, cfg):
    cluster_d = params.read_task_params("cluster")

    particles = pars["particles"]
    pileup = pars["pileup"]
    radius = pars["radius"]

    cl_size_radius = "{}_radius_{}".format(
        cfg["clusterStudies"]["clusterSizeBaseName"],
        str(round(radius, 3)).replace(".", "p"),
    )
    cluster_d["ClusterOutPlot"], cluster_d["ClusterOutValidation"] = cl_size_radius, cl_size_radius+"_valid"
    cluster_d["CoeffA"] = [radius] * 50
    
    if "weights" in cfg:
        cluster_d["weights"] = cfg["weights"]

    for key in ("ClusterInTC", "ClusterInSeeds", "ClusterOutPlot", "ClusterOutValidation"):
        name = cluster_d[key]

        cluster_d[key] =  "{}_{}_{}_posEta_9oct".format(particles, pileup, name)
    
    nevents_end = tasks.cluster.cluster_default(pars, **cluster_d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--radius", help="Coefficient to use as the max cluster radius", required=True, type=float)
    parser.add_argument("--particles", choices=("photons", "electrons", "pions"), required=True)
    parser.add_argument("--pileup", help="tag for PU200 vs PU0", choices=("PU0", "PU200"), required=True)
    parser.add_argument("--weighted", help="Apply pre-calculated layer weights", default=False)
    parsing.add_parameters(parser)
    
    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    radius_str = round(pars.radius, 3)

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    if pars.weighted:
        weight_dir = "{}/PU0/".format(params.LocalStorage)
        weights_by_particle = cl_helpers.read_weights(weight_dir, cfg)
        weights = weights_by_particle[pars.particles][radius_str]
        cfg["weights"] = weights

    cluster_radius(pars, cfg)