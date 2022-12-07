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

from bye_splits import utils
from utils import params

import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from progress.bar import IncrementalBar

import re
from copy import deepcopy

def binConv(vals, dist, amin):
    """
    Converts bin indexes back to values (central values in the bin).
    Assumes equally-spaced bins.
    """
    return (vals*dist) + (dist/2) + amin

def calcRzFromEta(eta):
    """R/z = arctan(theta) [theta is obtained from pseudo-rapidity, eta]"""
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

    df = df.drop(['subdet'], axis=1)
    return df, subdetCond

def get_html_name(script_name, name=''):
    f = Path(script_name).absolute().parents[1] / 'out'
    f /= name + '.html'
    return f

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

def dict_per_file(pars,file):
    # This will need to be changed eventually
    addit = re.split('gen_cl3d_tc_|_ThresholdDummy',file)[1]

    file_pars = {'opt': deepcopy(pars.opt_kw),
                 'fill': deepcopy(pars.fill_kw),
                 'smooth': deepcopy(pars.smooth_kw),
                 'seed': deepcopy(pars.seed_kw),
                 'cluster': deepcopy(pars.cluster_kw),
                 'validation': deepcopy(pars.validation_kw),
                 'energy': deepcopy(pars.energy_kw)}

    # Optimization pars
    file_pars['opt']['InFile'] = '{}.hdf5'.format(file)
    file_pars['opt']['OptIn'] = '{}_{}'.format(pars.opt_kw['OptIn'],addit)
    file_pars['opt']['OptEnResOut'] = '{}_{}'.format(pars.opt_kw['OptEnResOut'],addit)
    file_pars['opt']['OptPosResOut'] = '{}_{}'.format(pars.opt_kw['OptPosResOut'],addit)
    file_pars['opt']['OptCSVOut'] = '{}_{}'.format(pars.opt_kw['OptCSVOut'],addit)
    pars.set_dictionary(file_pars['opt'])

    # Fill pars
    file_pars['fill']['FillIn'] = file
    file_pars['fill']['FillOutPlot'] = '{}_{}'.format(pars.fill_kw['FillOutPlot'],addit)
    file_pars['fill']['FillOutComp'] = '{}_{}'.format(pars.fill_kw['FillOutComp'],addit)
    file_pars['fill']['FillOut'] = '{}_{}'.format(pars.fill_kw['FillOut'],addit)
    pars.set_dictionary(file_pars['fill'])

    # Smooth pars
    file_pars['smooth']['SmoothIn'] = '{}_{}'.format(pars.fill_kw['FillOut'],addit)
    file_pars['smooth']['SmoothOut'] = '{}_{}'.format(pars.smooth_kw['SmoothOut'],addit)
    pars.set_dictionary(file_pars['smooth'])

    # Seed pars
    file_pars['seed']['SeedIn'] = '{}_{}'.format(pars.smooth_kw['SmoothOut'],addit)
    file_pars['seed']['SeedOut'] = '{}_{}'.format(pars.seed_kw['SeedOut'],addit)
    pars.set_dictionary(file_pars['seed'])

    # Cluster pars
    file_pars['cluster']['ClusterInTC'] = '{}_{}'.format(pars.fill_kw['FillOut'],addit)
    file_pars['cluster']['ClusterInSeeds'] = '{}_{}'.format(pars.seed_kw['SeedOut'],addit)
    file_pars['cluster']['ClusterOutPlot'] = '{}_{}'.format(pars.cluster_kw['ClusterOutPlot'],addit)
    file_pars['cluster']['ClusterOutValidation'] = '{}_{}'.format(pars.cluster_kw['ClusterOutValidation'],addit)
    file_pars['cluster']['EnergyOut'] = '{}_{}'.format(pars.cluster_kw['EnergyOut'],addit)
    file_pars['cluster']['GenPart'] = '{}_{}'.format(file,addit)
    file_pars['cluster']['File'] = file
    pars.set_dictionary(file_pars['cluster'])

    # Validation pars
    file_pars['validation']['ClusterOutValidation'] = '{}_{}'.format(pars.cluster_kw['ClusterOutValidation'],addit)
    file_pars['validation']['FillOutComp'] = '{}_{}'.format(pars.fill_kw['FillOutComp'],addit)
    file_pars['validation']['FillOut'] = '{}_{}'.format(pars.fill_kw['FillOut'],addit)
    pars.set_dictionary(file_pars['validation'])

    # Energy pars
    file_pars['energy']['ClusterIn'] = '{}_{}'.format(pars.cluster_kw['ClusterOutValidation'],addit)
    file_pars['energy']['EnergyIn'] = '{}_{}'.format(pars.cluster_kw['EnergyOut'],addit)
    file_pars['energy']['EnergyOut'] = '{}_{}'.format(pars.energy_kw['EnergyOut'],addit)
    file_pars['energy']['EnergyPlot'] = '{}_{}'.format(pars.energy_kw['EnergyPlot'],addit)
    file_pars['energy']['File'] = file

    return file_pars

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
