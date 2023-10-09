# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing

from data_handle.data_process import get_data_reco_chain_start

import argparse
import random

random.seed(10)
import sys

import yaml

def start_chain(pars, cfg):
    particles = cfg["selection"]["particles"]
    pileup = "PU0" if not cfg["clusterStudies"]["pileup"] else "PU200"
    reprocess = cfg["clusterStudies"]["reprocess"]
    nevents = cfg["clusterStudies"]["nevents"]
    tag = cfg["clusterStudies"]["parquetTag"]

    df_gen, df_cl, df_tc = get_data_reco_chain_start(
        particles=particles, nevents=nevents, reprocess=reprocess, tag=tag
    )

    df_gen_pos, df_gen_neg = df_gen[ df_gen.gen_eta > 0 ], df_gen[ df_gen.gen_eta < 0]
    df_cl_pos, df_cl_neg = df_cl[ df_cl.cl3d_eta > 0 ], df_cl[ df_cl.cl3d_eta < 0]
    df_tc_pos, df_tc_neg = df_tc[ df_tc.tc_eta > 0 ], df_tc[ df_tc.tc_eta < 0]

    for i in range(2):
        if i==1:
            df_gen, df_cl, df_tc = df_gen_neg, df_cl_neg, df_tc_neg
            eta_tag = "negEta"
            df_gen["gen_eta"], df_cl["cl3d_eta"], df_tc["tc_eta"], df_tc["tc_z"] = abs(df_gen.gen_eta), abs(df_cl.cl3d_eta), abs(df_tc.tc_eta), abs(df_tc.tc_z)
        else:
            df_gen, df_cl, df_tc = df_gen_pos, df_cl_pos, df_tc_pos
            eta_tag = "posEta"

        print(f"{particles}: {eta_tag}")

        fill_d = params.read_task_params("fill")
        #for key in ("FillOut", "FillOutComp", "FillOutPlot"):
        for key in ("FillOut", "FillOutGenCl", "FillOutTcAll"):
            name = fill_d[key]
            fill_d[key] = "{}_{}_{}_{}_9oct".format(particles, pileup, name, eta_tag)
        tasks.fill.fill(pars, df_gen, df_cl, df_tc, **fill_d)

        smooth_d = params.read_task_params("smooth")
        for key in ("SmoothIn", "SmoothOut"):
            name = smooth_d[key]
            smooth_d[key] =  "{}_{}_{}_{}_9oct".format(particles, pileup, name, eta_tag)
        tasks.smooth.smooth(pars, **smooth_d)

        seed_d = params.read_task_params("seed")
        for key in ("SeedIn", "SeedOut"):
            name = seed_d[key]
            seed_d[key] = "{}_{}_{}_{}_9oct".format(particles, pileup, name, eta_tag)
        tasks.seed.seed(pars, **seed_d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    cfg["selection"]["particles"] = "pions"
    start_chain(common.dot_dict(vars(FLAGS)), cfg)

    '''for particles in ("photons", "electrons", "pions"):
        cfg["selection"]["particles"] = particles
        start_chain(common.dot_dict(vars(FLAGS)), cfg)'''