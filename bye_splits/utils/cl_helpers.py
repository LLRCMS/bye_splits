import os
import sys
import re
import argparse

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

from bye_splits.utils import common, parsing, params

parser = argparse.ArgumentParser(description="Seeding standalone step.")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()


def get_last_version(name):
    """Takes a template path, such as '/full/path/to/my_file.ext' and returns the path to the latest version
    corresponding to '/full/path/to/my_file_vN.ext' where N is the latest version number in the directory.
    """
    base = os.path.basename(name)
    base, ext = os.path.splitext(base)
    dir = os.path.dirname(name)
    if os.path.exists:
        # pattern = rf"{base}_v(\d{ext})"
        pattern = r"{}_v(\d{})".format(base, ext)
        matches = [re.match(pattern, file) for file in os.listdir(dir)]
        version = max(
            [
                int(match.group(1).replace(ext, ""))
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


# def_k = 0.0


def closest(list, k=0.0):
    """Find the element of a list containing strings ['coef_{float_1}', 'coef_{float_2}', ...] which is closest to some float_i"""
    try:
        list = np.reshape(np.asarray(list), 1)
    except ValueError:
        list = np.asarray(list)
    if isinstance(k, str):
        k_num = float(re.split("coef_", k)[1].replace("p", "."))
    else:
        k_num = k
    id = (np.abs(list - k_num)).argmin()
    return list[id]


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


def get_output_files(cfg):
    """Accepts a configuration file containing the base directory, a file basename, local (Bool) and pileUp (Bool).
    Finds the full paths of the files created by cluster_size.py, and returns
    a dictionary corresponding to particles:[file_paths]."""

    output_files = {"photons": [], "pions": [], "electrons": []}
    template = os.path.basename(
        common.fill_path(cfg["clusterSize"]["fileBaseName"], **vars(FLAGS))
    )
    template = re.split("_", template)
    if cfg["dirs"]["local"]:
        base_path = cfg["dirs"]["localDir"]
    else:
        base_path = (
            params.EOSStorage(FLAGS.user, "data/PU0/")
            if not cfg["clusterSize"]["pileUp"]
            else params.EOSStorage(FLAGS.user, "data/PU200/")
        )
    for particles in output_files.keys():
        particle_dir = (
            base_path + particles + "/" if cfg["dirs"]["local"] else base_path
        )
        files = [re.split("_", file) for file in os.listdir(particle_dir)]
        for filename in files:
            if set(template).issubset(set(filename)):
                path = os.path.join(f"{particle_dir}{'_'.join(filename)}")
                with pd.HDFStore(path, "r") as File:
                    if len(File.keys()) > 0:
                        if ("photon" in filename) or ("photons" in filename):
                            output_files["photons"].append(path)
                        elif ("electron" in filename) or ("electrons" in filename):
                            output_files["electrons"].append(path)
                        else:
                            output_files["pions"].append(path)

        # Get rid of duplicates that the dictionary filling produced
        for key in output_files.keys():
            output_files[key] = list(set(output_files[key]))

    return output_files
