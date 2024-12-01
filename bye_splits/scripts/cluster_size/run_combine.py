# coding: utf-8

_all_ = ['split_dfs', 'combine_normalize', 'combine_files_by_coef', 'split_and_norm', 'combine_cluster']

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from utils import params, common, parsing
from data_handle.data_process import get_data_reco_chain_start

import argparse
import re
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

def split_dfs(cl_df):
    """Splits the dataframe created by cluster() into an
    unweighted dataframe and a weighted dataframe."""

    weighted_cols = [col for col in cl_df.keys() if "weighted" in col]
    weighted_cols += [col for col in cl_df.keys() if "layer" in col]
    original_cols = [col.replace("weighted_","") for col in weighted_cols]
    weighted_cols += ["event"]
    original_cols += ["event"]

    original_df = cl_df[original_cols]
    weighted_df = cl_df[weighted_cols].rename(dict(zip(weighted_cols, original_cols)), axis=1)

    return original_df, weighted_df

def combine_normalize(cl_df, gen_df, dRThresh):
    """Combines a cluster dataframe with an associated
    gen-level dataframe. Useful variables that may or may not
    be present are added to the combined dataframe before
    normalized energy and pt columns are added."""

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

def combine_files_by_coef(in_dir, file_pattern):
    """Combines .hdf5 cluster files of individual
    radii into one .hdf5 file containing dataframes
    for all radii."""

    files = [
        file for file in os.listdir(in_dir) if re.search(file_pattern, file) is not None and "valid" not in file
    ]
    coef_pattern = r"coef_0p(\d+)"
    out_path = common.fill_path(file_pattern, data_dir=in_dir)
    with pd.HDFStore(out_path, "w") as clusterSizeOut:
        print("\nCombining Files:\n")
        for file in tqdm(files):
            key = re.search(coef_pattern, file).group()
            with pd.HDFStore(in_dir + "/" + file, "r") as clSizeCoef:
                clusterSizeOut[key] = clSizeCoef["/data"]

def split_and_norm(df_cl, df_gen, dRthresh):
    o_df, w_df = split_dfs(df_cl)
    normed_df, normed_w_df = combine_normalize(o_df, df_gen, dRthresh), combine_normalize(w_df, df_gen, dRthresh)
    df_dict = {"original": normed_df,
                "weighted": normed_w_df}
    return pd.Series(df_dict)

def combine_cluster(cfg, **pars):
    """Originally designed to combine the files returned by cluster for each radii,
    and to normalize each by the gen_particle information. Now accepts an optional --file
    parameter to normalize this, skipping the combinination step."""

    input_file_path = pars["file"] if "file" in pars.keys() else None
    unweighted = pars["unweighted"] if "unweighted" in pars.keys() else False

    particles = cfg["particles"]
    nevents = pars.nevents

    if input_file_path is None:
        pileup = "PU0" if not cfg["clusterStudies"]["pileup"] else "PU200"

        basename = cfg["clusterStudies"]["combination"][pileup][particles]["basename"]
        sub_dir = cfg["clusterStudies"]["combination"][pileup][particles]["sub_dir"]

        dir = "{}/{}/{}".format(params.LocalStorage, pileup, sub_dir)

        combine_files_by_coef(dir, basename)

        cl_size_out = common.fill_path(basename, data_dir=dir)

    else:
        cl_size_out = input_file_path

    with pd.HDFStore(cl_size_out, mode="a") as clSizeOut:
        df_gen, _, _ = get_data_reco_chain_start(
            particles=particles, nevents=nevents, reprocess=False, tag = cfg["clusterStudies"]["parquetTag"]
        )
        if "negEta" in cl_size_out:
            df_gen = df_gen[ df_gen.gen_eta < 0 ]
            df_gen["gen_eta"] = abs(df_gen.gen_eta)
        else:
            df_gen = df_gen[ df_gen.gen_eta > 0 ]
        dRthresh = cfg["selection"]["deltarThreshold"]
        if input_file_path is not None:
            clSizeOut["data"] = split_and_norm(clSizeOut["data"], df_gen, dRthresh) if not unweighted else combine_normalize(clSizeOut["data"], df_gen, dRthresh)
        else:
            coef_keys = clSizeOut.keys()
            print("\nNormalizing Files:\n")
            for coef in tqdm(coef_keys):
                clSizeOut[coef] = split_and_norm(clSizeOut[coef], df_gen, dRthresh) if not unweighted else combine_normalize(clSizeOut[coef], df_gen, dRthresh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--file", type=str)
    parser.add_argument("--unweighted", action="store_true")
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    for particles in ("photons", "electrons", "pions"):
        cfg.update({"particles": particles})
        combine_cluster(cfg, pars)