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

def energy(pars, **kw):

    coefs = np.linspace(0.00,0.05)
    energies = np.array([0])

    # This will create a .hdf5 file containing the cluster energy information corresponding to each coefficient
    # By default this file is assumed to be there
    if kw['ReInit']:
        for coef in coefs:
            clust_params = params.cluster_kw
            clust_params['ForEnergy'] = True
            clust_params['CoeffA'] = (coef,0)*52 #28(EM)+24(FH+BH)

            cluster.cluster(pars, **clust_params)

            #mean_en = cluster.cluster(pars, **clust_params)
            #clusterin = common.fill_path(kw['ClusterIn'], **pars)

    '''
    energies = energies[1:]

    for i, (coef, en) in enumerate(zip(coefs, energies)):
        isnan = math.isnan(en)
        if not isinstance(en, float) or isnan:
            #print("\nEnergy: ", en)
            coefs = np.delete(coefs,i)
            energies = np.delete(energies,i)

    max = energies[-1]
    en_norm = [en/max for en in energies]

    one_line = np.full(coefs.shape,1.0)

    p = figure()
    #p.layout.update(title='$\\text{Mean Cluster Energy vs. Radius From Seed}',xaxis_title='$\\DeltaR$',yaxis_title='$<E>$')
    p.line(coefs,en_norm,color='blue',line_dash = 'solid')
    p.line(coefs,one_line,color='green',line_dash='dashed')
    p.title.text = 'Normalized Cluster Energy vs. Radius From Seed'
    p.xaxis.axis_label = 'Distance From Seed (x/z)'
    p.yaxis.axis_label = 'Mean Cluster Energy/Max Cluster Energy'
    show(p)'''



if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing
    from bokeh.plotting import figure, output_file, show
    from bokeh.util.compiler import TypeScript

    #output_file('TestingEnergy.html')

    parser = argparse.ArgumentParser(description='Clustering standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_')

    energy(vars(FLAGS), **params.energy_kw)
