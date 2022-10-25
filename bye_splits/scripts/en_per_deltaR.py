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

def get_val(arr):
    return next((i for i in arr if i is not None),0.0)

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
    with h5py.File(infile,mode='r') as InFile:

        max_coef_key = 'coef_' + str(end).replace('.','p')
        max_energies = [i[0] for i in list(InFile[max_coef_key]['block1_values'])] #list(InFile...) produces a list of shape (1,0) arrays

        for key in InFile.keys():
            if key.startswith('coef_'):
                coef = InFile[key]

                num_cells = [i[0] for i in list(coef['block0_values'])]
                energy = [i[0] for i in list(coef['block1_values'])]

                breakpoint()

                en_norm = [en/max for en,max in list(itertools.zip_longest(energy,max_energies,fillvalue=0.0))]

                normed_energies = np.append(normed_energies,en_norm)
                cell_nums = np.append(cell_nums,num_cells)

    one_line = np.full(coefs.shape,1.0)

    p = figure(title='Normalized Cluster Energy vs. Radius From Seed', y_range=(0.0,1.2),y_axis_label='Cluster Energy/Max Cluster Energy')

    p.line(coefs,normed_energies,color='blue',line_dash = 'solid')
    p.line(coefs,one_line,color='green',line_dash='dashed')

    p.extra_y_ranges = {'y2': Range1d(start = 0, end=90)}
    p.add_layout(LinearAxis(y_range_name= 'y2',axis_label='Number of Cells'), 'right')
    p.line(coefs,cell_nums,color='red',line_dash = 'solid', y_range_name='y2')

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

    output_file('NewFigure.html')

    parser = argparse.ArgumentParser(description='Clustering standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

    energy(vars(FLAGS), **params.energy_kw)
