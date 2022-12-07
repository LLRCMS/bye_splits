import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits import utils
from bye_splits import tasks
from bye_splits.utils import common
from bye_splits.tasks import cluster
from bye_splits.tasks import cluster_test

import re
import numpy as np
import pandas as pd
import h5py

import math
import itertools

from dash import Dash, dcc, html, Input, Output, callback
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
from bye_splits.utils import params, parsing
import matplotlib.pyplot as plt
from matplotlib import cm

parser = argparse.ArgumentParser(description='Clustering standalone step.')
parsing.add_parameters(parser)
FLAGS = parser.parse_args()
assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

FLAGS.reg = 'All'
FLAGS.sel = 'below_eta_2.7'

input_files = params.fill_kw['FillInFiles']

# THIS FILE IS A WORK IN PROGRESS

dash.register_page(__name__, title='Energy', name='Energy')

layout = html.Div([
    html.H4('Normalized Cluster Energy'),

    html.Hr(),

    dcc.Graph(id='cl-en-graph',mathjax=True),

    html.P("EtaRange:"),
    dcc.RangeSlider(id='eta_range',min=1.4,max=2.7,step=0.1,value=[1.4,2.7]),

    html.P("Normalization:"),
    dcc.Dropdown(['Energy', 'PT'], 'Energy', id='normby')
])

@callback(
    Output("cl-en-graph", "figure"),
    Input("normby", "value"),
    Input("eta_range", "value"))

def plot_norm(normby, eta_range, pars=vars(FLAGS), init_files=input_files):
    plot_dict = {}
    normed_energies = dict.fromkeys(init_files.keys(),[0.0]) # Initialize at 0 since we only consider coefs[1:] (coefs[0] is an empty dataframe)
    start = params.energy_kw['EnergyOut']
    for key in init_files.keys():
        plot_dict[key] = [start+re.split('gen_cl3d_tc',file)[1] for file in init_files[key]]
        plot_dict[key] = [common.fill_path(file,**pars) for file in plot_dict[key]]
        if len(plot_dict[key])==1:
            with pd.HDFStore(plot_dict[key][0],'r') as File:
                coef_strs = File.keys()
                if normby=='max':
                    max = File[coef_strs[-1]].set_index('event').drop(columns=['matches','en_max'])
                for coef in coef_strs[1:]:
                    df = File[coef].set_index('event').drop(columns=['matches', 'en_max'])
                    if normby=='max':
                        df = df.join(max, on='event',rsuffix='_max')
                        df['normed_energies'] = df['en']/df['en_max']
                    elif normby=='pt':
                        df['normed_energies'] = df['cl3d_pt']/df['genpart_pt']
                    else:
                        df['normed_energies'] = df['en']/df['genpart_energy']

                    df = df[ df['genpart_exeta'] > eta_range[0] ]
                    df = df[ df['genpart_exeta'] < eta_range[1] ]

                    mean_energy = df['normed_energies'].mean()

                    normed_energies[key] = np.append(normed_energies[key],mean_energy)
        else:
            file_list = [pd.HDFStore(val,'r') for val in plot_dict[key]]
            coef_strs = file_list[0].keys()
            if normby=='max':
                max = pd.concat([file_list[i][coef_strs[-1]].set_index('event').drop(columns=['matches','en_max']) for i in range(len(file_list))])
            for coef in coef_strs[1:]:
                df_list = [file_list[i][coef] for i in range(len(file_list))]

                full_df = pd.concat(df_list)
                full_df = full_df.set_index('event').drop(columns=['matches', 'en_max'])

                if normby=='max':
                    full_df = full_df.join(max,rsuffix='_max')
                    full_df['normed_energies'] = full_df['en']/full_df['en_max']
                elif normby=='pt':
                    full_df['normed_energies'] = full_df['cl3d_pt']/full_df['genpart_pt']
                else:
                    full_df['normed_energies'] = full_df['en']/full_df['genpart_energy']

                full_df = full_df[ full_df['genpart_exeta'] > eta_range[0] ]
                full_df = full_df[ full_df['genpart_exeta'] < eta_range[1] ]

                mean_energy = full_df['normed_energies'].mean()

                normed_energies[key] = np.append(normed_energies[key], mean_energy)

            for file in file_list:
                file.close()

    start, end, tot = params.energy_kw['Coeffs']
    coefs = np.linspace(start, end, tot)

    one_line = np.full(tot,1.0)

    coef_ticks = coefs[0::5]

    coef_labels = [round(coef,3) for coef in coefs]
    coef_labels= coef_labels[0::5]

    color_list = cm.rainbow(np.linspace(0,1,len(normed_energies)))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

    en_phot = normed_energies['photon']
    en_pion = normed_energies['pion']

    fig.add_trace(go.Scatter(x=coefs, y=en_phot, name='Photon'), row=1, col=1)
    fig.add_trace(go.Scatter(x=coefs, y=en_pion, name='Pion'), row=1, col=2)

    fig.add_hline(y=1.0, line_dash='dash', line_color='green')

    fig.update_xaxes(title_text='Radius (Coeff)')

    fig.update_layout(title_text='Normalized Energy Distribution', yaxis_title_text=r'$\frac{\bar{E_{Cl}}}{\bar{E}_{Gen}}$')

    return fig
