# coding: utf-8

_all_ = ['start_chain']

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing
from data_handle.data_process import get_data_reco_chain_start

import argparse
import random
import yaml

def start_chain(pars, cfg):
    """Runs the first three steps of the TPG on
    negative and positive eta samples."""

    particles = cfg["selection"]["particles"]
    reprocess = cfg["clusterStudies"]["reprocess"]
    tag       = cfg["clusterStudies"]["parquetTag"]
    nevents   = pars.nevents
    pileup    = pars.pileup

    df_gen, df_cl, df_tc = get_data_reco_chain_start(
        particles=particles, nevents=nevents, reprocess=reprocess, tag=tag
    )

    df_gen_pos, df_gen_neg = df_gen[ df_gen.gen_eta > 0 ], df_gen[ df_gen.gen_eta < 0 ]
    df_cl_pos, df_cl_neg   = df_cl[ df_cl.cl3d_eta > 0 ],  df_cl[ df_cl.cl3d_eta < 0 ]
    df_tc_pos, df_tc_neg   = df_tc[ df_tc.tc_eta > 0 ],    df_tc[ df_tc.tc_eta < 0 ]

    eta_dict = {"negEta": {"df_gen": df_gen_neg,
                           "df_cl": df_cl_neg,
                           "df_tc": df_tc_neg},
                "posEta": {"df_gen": df_gen_pos,
                           "df_cl": df_cl_pos,
                           "df_tc": df_tc_pos}  
                }

    for eta_tag, dfs in eta_dict.items():
        df_gen, df_cl, df_tc = dfs.values()

        if eta_tag=="negEta":
            df_gen["gen_eta"], df_cl["cl3d_eta"], df_tc["tc_eta"], df_tc["tc_z"] = abs(df_gen.gen_eta), abs(df_cl.cl3d_eta), abs(df_tc.tc_eta), abs(df_tc.tc_z)

        print(f"{particles}: {eta_tag}")

        fill_d = params.read_task_params("fill")
        for key in ("FillOut", "FillOutGenCl", "FillOutTcAll"):
            name = fill_d[key]
            fill_d[key] = "{}_{}_{}_{}".format(particles, pileup, name, eta_tag)
        tasks.fill.fill(pars, df_gen, df_cl, df_tc, **fill_d)

        smooth_d = params.read_task_params("smooth")
        for key in ("SmoothIn", "SmoothOut"):
            name = smooth_d[key]
            smooth_d[key] =  "{}_{}_{}_{}".format(particles, pileup, name, eta_tag)
        tasks.smooth.smooth(pars, **smooth_d)

        seed_d = params.read_task_params("seed")
        for key in ("SeedIn", "SeedOut"):
            name = seed_d[key]
            seed_d[key] = "{}_{}_{}_{}".format(particles, pileup, name, eta_tag)
        tasks.seed.seed(pars, **seed_d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pileup", help="tag for pileup choice", default="PU0")
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    for particles in ("photons", "electrons", "pions"):
        cfg["selection"]["particles"] = particles
        start_chain(common.dot_dict(vars(FLAGS)), cfg)