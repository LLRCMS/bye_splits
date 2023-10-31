# coding: utf-8

_all_ = ['cluster_radius']

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing, cl_helpers
from utils.job_helpers import Arguments

import numpy as np
import pandas as pd
import yaml

arg_dict = {
    "--radius": {
        "help": "Coefficient to use as the max cluster radius",
        "required": True,
        "type": float
    },
    "--particles": {
        "choices": ("photons", "electrons", "pions"),
        "required": True
    }, 
    "--pileup": {
        "help": "tag for PU200 vs PU0",
        "choices": ("PU0", "PU200"),
        "required": True
    },
    "--weighted": {
        "help": "Apply pre-calculated layer weights",
        "action": "store_true"
    }
}

def cluster_radius(pars, cfg):
    """Runs the default clustering algorithm using the
    specified radius. Runs on both negative and positive
    eta files, and adds layer weights to the cluster
    kwargs if specified."""
    
    cluster_d = params.read_task_params("cluster")

    particles = pars["particles"]
    pileup = pars["pileup"]
    radius = pars["radius"]

    cl_size_radius = "{}_radius_{}".format(
        cfg["clusterStudies"]["clusterSizeBaseName"],
        str(round(radius, 3)).replace(".", "p"),
    )
    cluster_d["ClusterOutPlot"], cluster_d["ClusterOutValidation"] = cl_size_radius, cl_size_radius+"_valid"
    cluster_d["CoeffA"] = [radius] * (cfg["geometry"]["nlayersCEE"]+cfg["geometry"]["nlayersCEH"]) # Radii in each of the HGCAL layers
    
    if "weights" in cfg:
        cluster_d["weights"] = cfg["weights"]

    for eta_tag in ("negEta", "posEta"):
        for key in ("ClusterInTC", "ClusterInSeeds", "ClusterOutPlot", "ClusterOutValidation"):
            name = cluster_d[key]

            cluster_d[key] =  "{}_{}_{}_{}".format(particles, pileup, name, eta_tag)
        
        nevents_end = tasks.cluster.cluster_default(pars, **cluster_d)

if __name__ == "__main__":

    args = Arguments(script=__file__)
    FLAGS = args.add_args(description="Cluster size script.", arg_dict=arg_dict)
    pars = common.dot_dict(FLAGS)

    radius_str = round(pars.radius, 3)

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    if pars.weighted:
        weight_dir = os.path.join(params.LocalStorage, "PU0/")
        weights_by_particle = cl_helpers.read_weights(weight_dir, cfg)
        weights = weights_by_particle[pars.particles][radius_str]
        cfg["weights"] = weights

    cluster_radius(pars, cfg)