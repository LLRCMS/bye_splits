import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from bye_splits.utils import common, params

import numpy as np
import pandas as pd
import re
import argparse

def get_last_version(name):
    """Takes a template path, such as '/full/path/to/my_file.ext' and returns the path to the latest version
    corresponding to '/full/path/to/my_file_vN.ext' where N is the latest version number in the directory.
    """
    base = os.path.basename(name)
    base, ext = os.path.splitext(base)
    dir = os.path.dirname(name)
    if os.path.exists:
        pattern = r"{}_v(\d){}".format(base, ext)
        matches = [re.match(pattern, file) for file in os.listdir(dir)]
        version = max(
            [
                int(match.group(1))
                for match in matches
                if not match is None
            ]
        )
    return version

def update_version_name(name):
    """Takes the same template path as get_last_version(), and uses it to update the version number."""
    base, ext = os.path.splitext(name)
    version = 0 if not os.path.exists(name) else get_last_version(name)
    return f"{base}_v{str(version+1)}{ext}"

def closest(coef_list, k=0.0):
    """Find the element of a list containing strings ['coef_{float_1}', 'coef_{float_2}', ...] which is closest to some float_i"""
    coef_list = np.asarray(coef_list)
    if isinstance(k, str):
        k_num = float(re.split("coef_", k)[1].replace("p", "."))
    else:
        k_num = k
    id = (np.abs(coef_list - k_num)).argmin()
    return coef_list[id]

def get_str(coef, df_dict):
    """Accepts a coefficient, either as a float or string starting with coef_, along with a dictionary of coefficient:DataFrame pairs.
    Returns the coefficient string in the dictionary that is the closest to the passed coef.
    """
    if not isinstance(coef, str):
        coef_str = "coef_{}".format(str(coef).replace(".", "p"))
    else:
        coef_str = coef
    if coef_str not in df_dict.keys():
        coef_list = [
            float(re.split("coef_", key)[1].replace("p", ".")) for key in df_dict.keys()
        ]
        new_coef = closest(coef_list, coef)
        coef_str = "/coef_{}".format(str(new_coef).replace(".", "p"))
    return coef_str

# Old Naming Conventions used different column names in the dataframes
column_matching = {
    "etanew": "eta",
    "phinew": "phi",
    "genpart_exphi": "gen_phi",
    "genpart_exeta": "gen_eta",
    "genpart_energy": "gen_en",
}


def get_dfs(init_files, coef):
    """Takes a dictionary of input files (keys corresponding to particles, values corresponding to file paths containing DataFrames by coefficient), with a desired coefficient.
    Returns a new dictionary with the same keys, whose values correspond to the DataFrame of that particular coefficient.
    """
    df_dict = dict.fromkeys(init_files.keys(), [0.0])

    for key in init_files.keys():
        if len(init_files[key]) == 0:
            continue
        elif len(init_files[key]) == 1:
            file = pd.HDFStore(init_files[key][0], "r")

            if not isinstance(coef, str):
                coef = get_str(coef, file)

            df = file[coef]
        else:
            file_list = [pd.HDFStore(val, "r") for val in init_files[key]]
            if not isinstance(coef, str):
                coef = get_str(coef, file_list[0])
            df_list = [file_list[i][coef] for i in range(len(file_list))]
            df = pd.concat(df_list)
        df.rename(column_matching, axis=1, inplace=True)
        df_dict[key] = df

    return df_dict


def get_keys(init_files):
    """Returns the list of exact coefficient keys in the initial files; they're the same for all files, so we only need to check one."""
    file_path = init_files["photons"][0]

    with pd.HDFStore(file_path, "r") as file:
        keys = file.keys()

    return keys


def get_input_files(base_path, pile_up=False):
    """Accepts a base bath corresponding to the user's data directory, and returns a dictionary corresponding to particles:[root_files].
    Note that currently PU0 zero samples are stored in
    /eos/user/b/bfontana/FPGAs/new_algos/ and PU200 are stored in
    /eos/user/i/iehle/data/PU200/"""

    input_files = {"photons": [], "pions": [], "electrons": []}

    for key in input_files.keys():
        if not pile_up:
            path = (
                "{}{}/".format(f"{base_path}PU0/", key)
                if not "bfontana" in base_path
                else base_path
            )
            file = f"skim_small_{key}_0PU_bc_stc_hadd.root"
        else:
            path = (
                "{}{}/".format(f"{base_path}PU200/", key)
                if not "bfontana" in base_path
                else base_path
            )
            file = f"skim_{key}_hadd.root"

        input_files[key] = "{}{}".format(path, file)

    return input_files

def read_weights(dir, cfg, version="layer", mode="weights"):
    weights_by_particle = {}
    weight_path_templates = cfg["clusterStudies"]["weights"][version]
    for particle, basename in weight_path_templates.items():
        
        particle_dir = os.path.join(dir, particle, cfg["clusterStudies"]["weights"]["subDir"])

        files = [f for f in os.listdir(particle_dir) if basename in f]
        weights_by_radius = {}
        for file in files:
            radius = float(file.replace(".hdf5","").replace(f"{basename}_","").replace("r0","0").replace("p","."))
            infile = particle_dir+file
            with pd.HDFStore(infile, "r") as optWeights:
                weights_by_radius[radius] = optWeights[mode]
    
        weights_by_particle[particle] = weights_by_radius
    
    '''Weights are calculated from pt_norm distributions, which
    are distorted by brem events for electrons. As this is
    a physics effect uncorrelated to the TPG response, we correct
    electrons with weights derived from photon pt_norm distributions'''
    weights_by_particle["electrons"] = weights_by_particle["photons"]
    
    return weights_by_particle