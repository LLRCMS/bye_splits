import os
import sys
import re
import argparse

parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

from bye_splits.utils import common, parsing, params

parser = argparse.ArgumentParser(description='Seeding standalone step.')
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

def get_last_version(name):
    base = os.path.basename(name)
    base, ext = os.path.splitext(base)
    dir = os.path.dirname(name)
    if os.path.exists:
        pattern = fr"{base}_v(\d{ext})"
        matches = [re.match(pattern, file) for file in os.listdir(dir)]
        version = max([int(match.group(1).replace(ext,'')) for match in matches if not match is None])
    return version


def update_version_name(name):
    base, ext = os.path.splitext(name)
    version = 0 if not os.path.exists(name) else get_last_version(name)
    return f"{base}_v{str(version+1)}{ext}"

# Find the element of a list containing strings ['coef_{float_1}', 'coef_{float_2}', ...] which is closest to some float_i
def_k = 0.0
def closest(list, k=def_k):
    try:
        list = np.reshape(np.asarray(list), 1)
    except ValueError:
        list = np.asarray(list)
    if isinstance(k, str):
        k_num = float(re.split('coef_',k)[1].replace('p','.'))
    else:
        k_num = k
    id = (np.abs(list-k_num)).argmin()
    return list[id]

def get_str(coef, file):
    if not isinstance(coef, str):
        coef_str = 'coef_{}'.format(str(coef).replace('.','p'))
    else:
        coef_str = coef
    if coef_str not in file.keys():
        coef_list = [float(re.split('coef_',key)[1].replace('p','.')) for key in file.keys()]
        new_coef = closest(coef_list, coef)
        coef_str = '/coef_{}'.format(str(new_coef).replace('.','p'))
    return coef_str

def get_dfs(init_files, coef):
    df_dict = dict.fromkeys(init_files.keys(),[0.0])

    for key in init_files.keys():
        if len(init_files[key])==0:
            continue
        elif len(init_files[key])==1:

            file = pd.HDFStore(init_files[key][0],'r')
            
            if not isinstance(coef, str):
                coef = get_str(coef, file)
            
            df = file[coef]
        else:
            file_list = [pd.HDFStore(val,'r') for val in init_files[key]]
            if not isinstance(coef, str):
                coef = get_str(coef, file_list[0])
            df_list = [file_list[i][coef] for i in range(len(file_list))]
            df = pd.concat(df_list)
        df_dict[key] = df
    
    return df_dict

def get_keys(init_files):

    file_path = init_files['photons'][0]
    
    with pd.HDFStore(file_path, 'r') as file:
        keys = file.keys()

    return keys

def get_input_files(base_path, pile_up=False):
    """Note that currently PU0 zero samples are stored in
    /eos/user/b/bfontana/FPGAs/new_algos/ and PU200 are stored in
    /eos/user/i/iehle/data/PU200/"""

    input_files = {'photons':[], 'pions': [], 'electrons': []}
    
    for key in input_files.keys():
        if not pile_up:
            path = "{}{}/".format(f"{base_path}PU0/", key) if not 'bfontana' in base_path else base_path
            file = f"skim_small_{key}_0PU_bc_stc_hadd.root"
        else:
            path = "{}{}/".format(f"{base_path}PU200/", key) if not 'bfontana' in base_path else base_path
            file = f"skim_{key}_hadd.root"
        
        input_files[key] = "{}{}".format(path, file)
    
    return input_files

def get_output_files(cfg):
    output_files = {'photons':[], 'pions': [], 'electrons': []}
    template = os.path.basename(common.fill_path(cfg['ClusterSize']['FileBaseName'], **vars(FLAGS)))
    template = (re.split('_', template))
    if cfg['dirs']['Local']:
        base_path = cfg['dirs']['LocalDir']
    else:
        base_path = params.EOSStorage(FLAGS.user, 'data/PU0/') if not cfg['ClusterSize']['PileUp'] else params.EOSStorage(FLAGS.user, 'data/PU200/')
    for particles in output_files.keys():
        particle_dir = base_path+particles+'/' if cfg['dirs']['Local'] else base_path
        files = [re.split('_', file) for file in os.listdir(particle_dir)]
        for filename in files:
            if set(template).issubset(set(filename)):
                path = os.path.join(f"{particle_dir}{'_'.join(filename)}")
                with pd.HDFStore(path, "r") as File:
                    if len(File.keys())>0:
                        if ('photon' in filename) or ('photons' in filename):
                            output_files['photons'].append(path)
                        elif ('electron' in filename) or ('electrons' in filename):
                            output_files['electrons'].append(path)
                        else:
                            output_files['pions'].append(path)
    
    return output_files