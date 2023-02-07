# coding: utf-8

_all_ = [ ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing

from bye_splits.iterative_optimization import optimization

import csv
import argparse
import random; random.seed(10)
import numpy as np
import pandas as pd
import sys
import re

# ASK LOUIS/BRUNO ABOUT ALGO (also: bc_stc = best choice super trigger cell)
pion_base_path = "/data_CMS_upgrade/sauvan/HGCAL/2210_Ehle_clustering-studies/SinglePion_PT0to200/PionGun_Pt0_200_PU0_HLTSummer20ReRECOMiniAOD_2210_clustering-study_v3-29-1/221018_121053/ntuple_"
files = {'photons': ['/data_CMS/cms/alves/L1HGCAL/skim_small_photons_0PU_bc_stc_hadd.root'], 'pions': [f"{pion_base_path}{i+1}.root" for i in range(3)]}
gen_trees = {'photons': 'FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple/HGCalTriggerNtuple', 'pions':'hgcalTriggerNtuplizer/HGCalTriggerNtuple'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--process',
                        help='reprocess trigger cell geometry data',
                        action='store_true')
    parser.add_argument('-p', '--plot',
                        help='plot shifted trigger cells instead of originals',
                        action='store_true')
    parser.add_argument('--no_fill',    action='store_true')
    parser.add_argument('--no_smooth',  action='store_true')
    parser.add_argument('--no_seed',    action='store_true')
    parser.add_argument('--no_cluster', action='store_true')
    nevents_help = "Number of events for processing. Pass '-1' for all events."
    parser.add_argument('-n', '--nevents', help=nevents_help,
                        default=-1, type=int)
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()
    FLAGS.reg = 'All'
    

    pars_d = {'sel'           : FLAGS.sel,
              'reg'           : FLAGS.reg,
              'seed_window'   : FLAGS.seed_window,
              'smooth_kernel' : FLAGS.smooth_kernel,
              'cluster_algo'  : FLAGS.cluster_algo ,
              'ipar'          : FLAGS.ipar}

    tc_map = optimization(pars_d, **params.opt_kw)

    for particle, paths in files.items():
        for file_path in paths:
            # Get file addition
            file = os.path.basename(file_path).replace(".root", "")

            outcsv = common.fill_path('{}_{}'.format(params.opt_kw['OptCSVOut'],file), ext='csv', **pars_d)
            outresen  = common.fill_path('{}_{}'.format(params.opt_kw['OptEnResOut'],file),  **pars_d)
            outrespos = common.fill_path('{}_{}'.format(params.opt_kw['OptPosResOut'],file), **pars_d)

            with open(outcsv, 'w', newline='') as csvfile, pd.HDFStore(outresen, mode='w') as storeEnRes, pd.HDFStore(outrespos, mode='w') as storePosRes:
                fieldnames = ['ipar', 'c_loc1', 'c_loc2', 'c_rem1', 'c_rem2',
                            'locrat1', 'locrat2', 'remrat1', 'remrat2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                sys.stderr.flush()

                # Initiate file specific parameters
                file_pars = common.FileDict(params,file_path)

                if not FLAGS.no_fill:
                    print("Starting filling step.")
                    tasks.fill.fill(pars_d, FLAGS.nevents, tc_map, **file_pars.get_fill_pars())
                    print("Finished filling step.")

                if not FLAGS.no_smooth:
                    print("Starting smoothing step.")
                    tasks.smooth.smooth(pars_d, **file_pars.get_smooth_pars())
                    print("Finished smoothing step.")

                if not FLAGS.no_seed:
                    print("Starting seeding step.")
                    tasks.seed.seed(pars_d, **file_pars.get_seed_pars())
                    print("Finished seeding step.")

                if not FLAGS.no_cluster:
                    print("Starting clustering step.")
                    energy_kw = file_pars.get_energy_pars()
                    cluster_kw = file_pars.get_cluster_pars()

                    if energy_kw['ReInit'] == True:
                        cluster_kw['ForEnergy'] = True
                        start, end, tot = energy_kw['Coeffs']
                        coefs = np.linspace(energy_kw['Coeffs'])
                        for coef in coefs:
                            print(f"\nStarting Coefficient: {coef}")
                            cluster_kw['CoeffA'] = (coef, 0)*50
                            tasks.cluster.cluster(pars_d, **cluster_kw)
                    else:
                        tasks.cluster.cluster(pars_d, **cluster_kw)
                    
                    print("Finished clustering step.")