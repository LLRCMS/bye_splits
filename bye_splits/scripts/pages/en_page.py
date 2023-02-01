import os
import sys

#parent_dir = os.path.abspath(__file__ + 4 * '/..')
parent_dir = '/eos/user/i/iehle/data/PU0/'

sys.path.insert(0, parent_dir)

from bye_splits.utils import common

import re
import numpy as np
import pandas as pd

from dash import dcc, html, Input, Output, callback
import dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
import bye_splits
from bye_splits.utils import params, parsing

parser = argparse.ArgumentParser(description='Clustering standalone step.')
parsing.add_parameters(parser)
FLAGS = parser.parse_args()
assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

FLAGS.reg = 'All'
FLAGS.sel = 'below_eta_2.7'

input_files = params.fill_kw['FillInFiles']

dash.register_page(__name__, title='Energy', name='Energy')

layout = html.Div([
    html.H4('Normalized Cluster Energy'),

    html.Hr(),

    dcc.Graph(id='cl-en-graph',mathjax=True),

    html.P("EtaRange:"),
    dcc.RangeSlider(id='eta_range',min=1.4,max=2.7,step=0.1,value=[1.6,2.7]),

    html.P("Normalization:"),
    dcc.Dropdown(['Energy', 'PT'], 'PT', id='normby')
])

def fill_dict_w_mean_norm(key, coef, eta, df, norm, out_dict):
    if norm=='max':
        df = df.join(max, on='event',rsuffix='_max')
        df['normed_energies'] = df['en']/df['en_max']
    elif norm=='PT':
        df['pt'] = df['en']/np.cosh(df['etanew'])
        df['normed_energies'] = df['pt']/df['genpart_pt']
    else:
        df['normed_energies'] = df['en']/df['genpart_energy']

    df = df[ df['genpart_exeta'] > eta[0] ]
    df = df[ df['genpart_exeta'] < eta[1] ]

    mean_energy = df['normed_energies'].mean()

    out_dict[key] = np.append(out_dict[key],mean_energy)    

def write_plot_file(input_files, norm, eta, outfile, pars=vars(FLAGS)):
    plot_dict = {}
    normed_energies = dict.fromkeys(input_files.keys(),[0.0]) # Initialize at 0 since we only consider coefs[1:] (coefs[0] is an empty dataframe)
    start = params.energy_kw['EnergyOut']
    for key in input_files.keys():
        plot_dict[key] = [start+re.split('gen_cl3d_tc',file)[1] for file in input_files[key]]
        plot_dict[key] = [common.fill_path(file,**pars) for file in plot_dict[key]]
        
        if len(plot_dict[key])==1:
            with pd.HDFStore(plot_dict[key][0],'r') as File:
                coef_strs = File.keys()
                if norm=='max':
                    max = File[coef_strs[-1]].set_index('event').drop(columns=['matches','en_max'])
                for coef in coef_strs[1:]:
                    df = File[coef].set_index('event').drop(columns=['matches', 'en_max'])
                    fill_dict_w_mean_norm(key, coef, eta, df, norm, normed_energies)

        else:
            file_list = [pd.HDFStore(val,'r') for val in plot_dict[key]]
            coef_strs = file_list[0].keys()
            if norm=='max':
                max = pd.concat([file_list[i][coef_strs[-1]].set_index('event').drop(columns=['matches','en_max']) for i in range(len(file_list))])
            for coef in coef_strs[1:]:
                df_list = [file_list[i][coef] for i in range(len(file_list))]
                full_df = pd.concat(df_list)

                full_df = full_df.set_index('event').drop(columns=['matches', 'en_max'])
                fill_dict_w_mean_norm(key, coef, eta, full_df, norm, normed_energies)

            for file in file_list:
                file.close()
    
    with pd.HDFStore(outfile, 'w') as PlotFile:
        PlotFile.put('Normed_Dist', pd.DataFrame.from_dict(normed_energies))
    
    return normed_energies


@callback(
    Output("cl-en-graph", "figure"),
    Input("normby", "value"),
    Input("eta_range", "value"))

def plot_norm(normby, eta_range, pars=vars(FLAGS), init_files=input_files, plot_file='normed_distribution'):
    global y_axis_title

    if normby=='Energy':
        y_axis_title = r'$\frac{\bar{E_{Cl}}}{\bar{E}_{Gen}}$'
    elif normby=='PT':
        y_axis_title = r'$\frac{\bar{p_T}^{Cl}}{\bar{p_T}^{Gen}}$'
    else:
        y_axis_title = r'$\frac{E_{Cl}}{E_{Max}}$'

    plot_filename = "{}{}_{}_eta_{}_{}.hdf5".format(parent_dir, normby, plot_file, eta_range[0], eta_range[1])

    if not os.path.exists(plot_filename):
        normed_dist = write_plot_file(init_files, normby, eta_range, plot_filename)
    else:
        try:
            with pd.HDFStore(plot_filename, "r") as PlotFile:
                normed_dist = PlotFile['/Normed_Dist'].to_dict(orient='list')
        except:
            os.remove(plot_filename)
            normed_dist = write_plot_file(init_files, normby, eta_range, plot_filename)

    start, end, tot = params.energy_kw['Coeffs']
    coefs = np.linspace(start, end, tot)

    coef_labels = [round(coef,3) for coef in coefs]
    coef_labels= coef_labels[0::5]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

    en_phot = normed_dist['photon']
    en_pion = normed_dist['pion']

    fig.add_trace(go.Scatter(x=coefs, y=en_phot, name='Photon'), row=1, col=1)
    fig.add_trace(go.Scatter(x=coefs, y=en_pion, name='Pion'), row=1, col=2)

    fig.add_hline(y=1.0, line_dash='dash', line_color='green')

    fig.update_xaxes(title_text='Radius (Coeff)')

    fig.update_layout(title_text='Normalized {} Distribution'.format(normby), yaxis_title_text=y_axis_title)

    return fig
