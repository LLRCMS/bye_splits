# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

from utils import params

import os
import yaml
import numpy as np
import pandas as pd
import re


def binConv(vals, dist, amin):
    """
    Converts bin indexes back to values (central values in the bin).
    Assumes equally-spaced bins.
    """
    return (vals * dist) + (dist / 2) + amin


def calcRzFromEta(eta):
    """R/z = tan(theta) [theta is obtained from pseudo-rapidity, eta]"""
    _theta = 2 * np.arctan(np.exp(-1 * eta))
    return np.tan(_theta)


class dot_dict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def fill_path(base_path, data_dir=params.LocalStorage, ext="hdf5", **kw):
    """Create unique file name base on user input parameters."""

    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
        base_path += '_' + cfg['base']['FesAlgo']
    
    def add_if_exists(s, prefix):
        nonlocal base_path
        if s in kw:
            base_path += "_" + prefix + "_" + str(kw[s]).replace(".", "p")

    strings = {
        "sel": "SEL",
        "reg": "REG",
        "seed_window": "SW",
        "smooth_kernel": "SK",
        "cluster_algo": "CA",
    }

    for k, v in strings.items():
        add_if_exists(k, v)

    base_path += "." + ext
    return os.path.join(data_dir, base_path)


class SupressSettingWithCopyWarning:
    """
    Temporarily supress pandas SettingWithCopyWarning.
    It is known to ocasionally provide false positives.
    https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    """

    def __init__(self, chained=None):
        acceptable = [None, "warn", "raise"]
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw


def get_detector_region_mask(df, region):
    """
    Obtain a mask to filter a specific detector region.
    subdetectors: - ECAL (1)
                  - HCAL silicon (2)
                  - HCAL scintillator (10)
    """
    if region == "Si":
        subdetCond = (df.subdet == 1) | (df.subdet == 2)
    elif region == "ECAL":
        subdetCond = df.subdet == 1
    elif region == "HCAL":
        subdetCond = (df.subdet == 2) | (df.subdet == 10)
    elif region == "MaxShower":
        subdetCond = (df.subdet == 1) & (df.layer >= 8) & (df.layer <= 15)
    elif region == "ExcludeMaxShower":
        subdetCond = (df.subdet == 1) & (df.layer < 8) | (df.layer > 15)

    df = df.drop(["subdet"], axis=1)
    return df, subdetCond

def print_histogram(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 0:
                print("-", end="|")
            else:
                print("X", end="|")
        print()

# Accepts a template for a full path to a file and increments the version
def increment_version(file_path):
    dir, file = os.path.split(file_path)
    base, ext = os.path.splitext(file)
    i = 0
    file_path = "{}/{}_v{}{}".format(dir, base, i, ext)
    while os.path.exists(file_path):
        i += 1
        file_path = "{}/{}_v{}{}".format(dir, base, i, ext)
    return file_path


# Grab the most recent version of the file corresponding to the template file_path (or return all matches)
def grab_most_recent(file_path, return_all=False):
    dir, file = os.path.split(file_path)
    base, ext = os.path.splitext(file)
    files = os.listdir(dir)
    version_pattern = re.compile("{}_v(\\d+)\\{}".format(base, ext))
    matches = [version_pattern.search(file) for file in files]
    matches = [match for match in matches if not match is None]
    if len(matches) > 0:
        matches = [int(match.group(1)) for match in matches]
        most_recent = max(matches)
        file_path = dir + "/" + base + "_v" + str(most_recent) + ext
        file_paths = [dir + "/" + base + "_v" + str(f) + ext for f in matches]
        if not return_all:            
            return file_path
        else:
            return file_paths
    else:
        return None

def compare_file_contents(file_path, buffer_list):
    """
    Compares the content in <file_path> with <buffer_list>,
    which should be a list of strings that you wish to write
    to a new file.
    """
    with open(file_path, "r") as file:
        contents = file.readlines()
    return contents==buffer_list

def write_file_version(template, version):
    file_name = increment_version(template)
    with open(file_name, "w") as job_file:
        job_file.writelines(version)
    st = os.stat(file_name)
    os.chmod(file_name, st.st_mode | 0o744)
    return file_name

def conditional_write(file_versions, file_template, current_version):
    """
    Loops through the files in <file_versions>, comparing their contents
    to the current version. If an identical version is found, the function
    breaks and does nothing. Otherwise, it will write the contents in
    <current_version> to an updated version number whose basename corresponds to
    <file_template>.
    """
    if file_versions != None:
        identical_version = False
        for file in file_versions:
            if not compare_file_contents(file, current_version):
                continue
            else:
                identical_version = True
                file_path = grab_most_recent(file_template)
                break
        if not identical_version:
            file_path = write_file_version(file_template, current_version)
    
    else:
        file_path = write_file_version(file_template, current_version)
    return file_path