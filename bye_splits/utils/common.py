# coding: utf-8

_all_ = [ ]

import os
from os import path as op
from pathlib import Path
import sys
import subprocess
from subprocess import Popen, PIPE

parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits import utils
from utils import params

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from progress.bar import IncrementalBar

import re
from copy import copy, deepcopy

def binConv(vals, dist, amin):
    """
    Converts bin indexes back to values (central values in the bin).
    Assumes equally-spaced bins.
    """
    return (vals*dist) + (dist/2) + amin

def calcRzFromEta(eta):
    """R/z = tan(theta) [theta is obtained from pseudo-rapidity, eta]"""
    _theta = 2*np.arctan( np.exp(-1 * eta) )
    return np.tan( _theta )

class dot_dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def fill_path(base_path, ext='hdf5', **kw):
    """Create unique file name base on user input parameters."""
    def add_if_exists(s, prefix):
        nonlocal base_path
        if s in kw:
            base_path += '_' + prefix + '_' + str(kw[s]).replace('.','p')

    strings = {'ipar'          : 'PAR',
               'sel'           : 'SEL',
               'reg'           : 'REG',
               'seed_window'   : 'SW',
               'smooth_kernel' : 'SK',
               'cluster_algo'  : 'CA'}

    for k,v in strings.items():
        add_if_exists(k, v)

    base_path += '.' + ext

    path = 'OutPath' if ext == 'html' else 'BasePath'
    return Path(params.base_kw[path]) / base_path

class SupressSettingWithCopyWarning:
    """
    Temporarily supress pandas SettingWithCopyWarning.
    It is known to ocasionally provide false positives.
    https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    """
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw

def get_column_idx(columns, col):
    return columns.index(col)

def get_detector_region_mask(df, region):
    """
    Obtain a mask to filter a specific detector region.
    subdetectors: - ECAL (1)
                  - HCAL silicon (2)
                  - HCAL scintillator (10)
    """
    if region == 'Si':
        subdetCond = (df.subdet == 1) | (df.subdet == 2)
    elif region == 'ECAL':
        subdetCond = (df.subdet == 1)
    elif region == 'HCAL':
        subdetCond = (df.subdet == 2) | (df.subdet == 10)
    elif region == 'All':
        subdetCond = (df.subdet == 1) | (df.subdet == 2) | (df.subdet == 10)
    elif region == 'MaxShower':
        subdetCond = ( (df.subdet == 1) &
                       (df.layer >= 8) & (df.layer <= 15) )
    elif region == 'ExcludeMaxShower':
        subdetCond = ( (df.subdet == 1) &
                       (df.layer < 8) | (df.layer > 15) )

    df = df.drop(['subdet'], axis=1)
    return df, subdetCond

def get_html_name(script_name, name=''):
    f = Path(script_name).absolute().parents[1] / 'out'
    f /= name + '.html'
    return f

def print_histogram(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i,j] == 0:
                print('-', end='|')
            else:
                print('X', end='|')
        print()

def tc_base_selection(df, region, pos_endcap, range_rz):
    if pos_endcap:
        df = df[ df.zside == 1 ] #only look at positive endcap
        df = df.drop(['zside'], axis=1)

    df['R'] = np.sqrt(df.x*df.x + df.y*df.y)
    df['Rz'] = df.R / abs(df.z)

    #the following cut removes almost no event at all
    df = df[ ((df['Rz'] < range_rz[1]) &
              (df['Rz'] > range_rz[0])) ]

    df, subdetCond = get_detector_region_mask(df, region)
    return df, subdetCond

def transform(nested_list):
    regular_list=[]
    for ele in nested_list:
        if type(ele) is list:
            regular_list.append(ele)
        else:
            regular_list.append([ele])
    return regular_list

class FileDict:
    def __init__(self, pars, file):
        self.pars = pars
        self.file = file
        self.init_pars = {'opt': copy(pars.opt_kw),
                          'fill': copy(pars.fill_kw),
                          'smooth': copy(pars.smooth_kw),
                          'seed': copy(pars.seed_kw),
                          'cluster': copy(pars.cluster_kw),
                          'validation': copy(pars.validation_kw),
                          'energy': copy(pars.energy_kw)}
        self.addit = os.path.basename(self.file).replace(".root", "")
        self.file_addit = lambda x: f"{x}_{self.addit}"

    def get_opt_pars(self):
        opt_kw = deepcopy(self.init_pars['opt'])
        opt_kw['InFile'] = f"{self.file}.hdf5"
        for key in opt_kw.keys():
            if key != 'InFile' and key not in params.base_kw:
                opt_kw[key] = self.file_addit(self.pars.opt_kw[key])
        return opt_kw

    def get_fill_pars(self):
        fill_kw = deepcopy(self.init_pars['fill'])
        fill_kw['FillIn'] = self.file.replace('.root', '')
        for key in fill_kw.keys():
            if key != 'FillIn' and key not in params.base_kw:
                fill_kw[key] = self.file_addit(self.pars.fill_kw[key])
        return fill_kw

    def get_smooth_pars(self):
        smooth_kw = deepcopy(self.init_pars['smooth'])
        for key in smooth_kw.keys():
            smooth_kw[key] = self.file_addit(self.pars.smooth_kw[key])
        return smooth_kw

    def get_seed_pars(self):
        seed_kw = deepcopy(self.init_pars['seed'])
        for key in seed_kw.keys():
            seed_kw[key] = self.file_addit(self.pars.seed_kw[key])
        return seed_kw
    
    def get_cluster_pars(self):
        cluster_kw = deepcopy(self.init_pars['cluster'])
        cluster_kw['File'] = self.file
        for key in cluster_kw.keys():
            if key != 'File' and key not in params.base_kw:
                cluster_kw[key] = self.file_addit(self.pars.cluster_kw[key])
        return cluster_kw

    def get_validation_pars(self):
        validation_kw = deepcopy(self.init_pars['validation'])
        for key in validation_kw.keys():
            validation_kw[key] = self.file_addit(self.pars.validation_kw[key])
        return validation_kw

    def get_energy_pars(self):
        energy_kw = deepcopy(self.init_pars['energy'])
        energy_kw['File'] = self.file
        for key in energy_kw.keys():
            if key != 'File' and key not in params.base_kw:
                energy_kw[key] = self.file_addit(self.pars.energy_kw[key])
        return energy_kw

def point_to_root_file(sample_list, dict):
    # Initialize paths
    xrd_door = 'root://polgrid4.in2p3.fr/'
    store = '/dpm/in2p3.fr/home/cms/trivcat/store/user/lportale/'

    def init_path_deviation(path1, path2):
        set1, set2 = OrderedSet(re.split(r'(/)', path1)), OrderedSet(re.split(r'(/)', path2))

        str_diff = set1.symmetric_difference(set2)

        return list(str_diff)

    def check_substr(substr, arr):
        val = False

        for file in arr:
            if substr in file:
                val = True
                break

        return val

    def recursive_path(init_dir):
        path = init_dir

        init_dir_list = re.split("\n", subprocess.run(['gfal-ls', path], text=True, capture_output=True).stdout)
        init_dir_list = [dir for dir in init_dir_list if len(dir)!=0]

        full_paths=[]
        count = 0
        while len(init_dir_list) > 0:
            if count == 0:
                dir_list = init_dir_list

            #As long as none of the files in dir_list contain .root, continue path and finding next directory
            while not check_substr(".root", dir_list):
                path += '/' + dir_list[0]
                dir_list = re.split("\n", subprocess.run(['gfal-ls', path], text=True, capture_output=True).stdout)
                dir_list = [dir for dir in dir_list if len(dir)!=0]

            for file in dir_list:
                file_path = path + '/' + file
                if file not in full_paths:
                    full_paths.append(file_path)

            init_path = init_path_deviation(init_dir, file_path)[0]
            if init_path in init_dir_list:
                init_dir_list.remove(init_path)
            if len(init_dir_list) > 0:
                path = init_dir + '/' + init_dir_list[0]
                count += 1
                dir_list = re.split("\n", subprocess.run(['gfal-ls', path], text=True, capture_output=True).stdout)
                dir_list = [dir for dir in dir_list if len(dir)!=0]
            else:
                break

        return full_paths

    for key, sample in zip(dict.keys(), sample_list):
        path = xrd_door+store+sample
        with_pu = [path for path in recursive_path(path) if "PU200" in path]
        dict[key] = with_pu
