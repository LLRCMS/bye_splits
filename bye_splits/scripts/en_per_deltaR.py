import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits import utils
from bye_splits import tasks
from bye_splits.utils import common
from bye_splits.tasks import cluster

import re
import numpy as np
import pandas as pd
import h5py

import math
import itertools

def energy(pars, **kw):

    start, end, tot = kw['Coeffs']
    coefs = np.linspace(start, end, tot)
    #energies = np.array([])
    normed_energies = np.array([])
    cell_nums = np.array([])

    # This will create a .hdf5 file containing the cluster energy information corresponding to each coefficient
    # By default this file is assumed to be there
    if kw['ReInit']:
        clust_params = params.cluster_kw
        clust_params['ForEnergy'] = True
        for coef in coefs:
            clust_params['CoeffA'] = (coef,0)*52 #28(EM)+24(FH+BH)
            cluster.cluster(pars, **clust_params)

    # Create list of mean cluster energy from cluster_energy... file
    infile = common.fill_path(kw['EnergyIn'], **pars)
    genFile = common.fill_path(kw['genFile'], **pars)
    outfile = common.fill_path('normed_energy', **pars)
    with pd.HDFStore(infile,'r') as InFile, pd.HDFStore(genFile,'r') as GenFile, pd.HDFStore(outfile,'w') as OutFile:

        coef_keys = ['/coef_' + str(coef).replace('.','p') for coef in coefs]

        #for coef in coef_keys[1:]:
        for coef in coef_keys:

            df_current = InFile[coef]
            df_final = InFile[coef_keys[-1]]

            df_current['seed_idx'] = df_current.index
            df_final['seed_idx'] = df_final.index

            combined_df = df_current.set_index(['event','seed_idx']).join(df_final.set_index(['event','seed_idx']), how='inner', lsuffix='_current', rsuffix='_max')

            combined_df['normed_en'] = combined_df['en_current']/combined_df['en_max']

            if coef == coef_keys[0]:
                mean_normed_en = 0.0
            else:
                mean_normed_en = combined_df['normed_en'].mean()

            normed_energies = np.append(normed_energies,mean_normed_en)

            OutFile[coef] = combined_df.drop(columns=['en_max','xnew_max','ynew_max','z_max','Rz_max','etanew_max','phinew_max','Ncells_max'])



    one_line = np.full(coefs.shape,1.0)

    p = figure(title='Normalized Cluster Energy vs. Radius From Seed', y_range=(0.0,1.2),y_axis_label='Cluster Energy/Max Cluster Energy')

    p.line(coefs,normed_energies,color='blue',line_dash = 'solid')
    p.line(coefs,one_line,color='green',line_dash='dashed')

    #p.extra_y_ranges = {'y2': Range1d(start = 0, end=90)}
    #p.add_layout(LinearAxis(y_range_name= 'y2',axis_label='Number of Cells'), 'right')
    #p.line(coefs,cell_nums,color='red',line_dash = 'solid', y_range_name='y2')

    #p.title.text = 'Normalized Cluster Energy vs. Radius From Seed'
    p.xaxis.axis_label = 'Distance From Seed (x/z)'
    #p.yaxis.axis_label = 'Cluster Energy/Max Cluster Energy'
    show(p)



if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing
    from bokeh.plotting import figure, output_file, show
    from bokeh.util.compiler import TypeScript
    from bokeh.models import LinearAxis, Range1d

    output_file('ClusterEnergy.html')

    parser = argparse.ArgumentParser(description='Clustering standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

    energy(vars(FLAGS), **params.energy_kw)
