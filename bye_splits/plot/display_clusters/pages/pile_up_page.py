import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * '/..')
sys.path.insert(0, parent_dir)

from bye_splits.utils import common

import re
import numpy as np
import pandas as pd
import uproot as up
import yaml

from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import dash
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
import bye_splits
from bye_splits.utils import params, parsing, cl_helpers

parser = argparse.ArgumentParser(description='')
parsing.add_parameters(parser)
FLAGS = parser.parse_args() 

phot_color = "#AFB8FF"
el_color = "#EDA2A8"
#phot_color = '#1f77b4'
#el_color = '#e377c2'

with open(params.CfgPaths['cluster_app'], 'r') as afile:
    cfgprod = yaml.safe_load(afile)

if cfgprod['dirs']['Local']:
    data_dir = cfgprod['dirs']['LocalDir']
else:
    data_dir = params.EOSStorage(FLAGS.user, 'data/')

input_files = cl_helpers.get_input_files(data_dir, pile_up=True)

tree_name = cfgprod['ClusterSize']['Tree']

common_branches = ['event', 'deltaR', 'matches']

branches = lambda tree, string : common_branches+[branch for branch in tree.keys() if string in branch]
parquet_name = lambda particle, string: f"{data_dir}PU200/{particle}/{string}_df.parquet"
grab_df = lambda tree, branches: tree.arrays(branches, library='pd')

def generate_dfs(input_files):
    df_dict = {}
    for key, val in input_files.items():
        if os.path.exists(val):
            with up.open(val) as rootFile:
                tree = rootFile[tree_name]

                tc_branches, genpart_branches, cl3d_branches = branches(tree, 'good_tc'), branches(tree, 'genpart'), branches(tree, 'cl3d')
                
                tc_outfile, genpart_outfile, cl3d_outfile = parquet_name(key, 'tc'), parquet_name(key, 'genpart'), parquet_name(key, 'cl3d')

                # Trigger Cell Branches are far too large to be read the same way as the other branches--it is far better to convert the trigger dfs for _each skimmed ntuple_ to parquet format first and then combine
                '''if not os.path.exists(tc_outfile):
                    tc_df = grab_df(tree, tc_branches)
                    tc_df.to_parquet(tc_outfile)
                else:
                    tc_df = pd.read_parquet(tc_outfile)'''
                
                if not os.path.exists(genpart_outfile):
                    genpart_df = grab_df(tree, genpart_branches)
                    genpart_df = pd.merge(genpart_df[0].drop('event', axis=1), genpart_df[1], on=['entry', 'subentry'])
                    genpart_df = genpart_df.rename({'good_genpart_exphi':'good_genpart_phi', 'good_genpart_exeta':'good_genpart_eta'}, axis=1)
                    genpart_df['good_genpart_pt'] = genpart_df['good_genpart_energy']/np.cosh(genpart_df['good_genpart_eta'])
                    genpart_df.to_parquet(genpart_outfile)
                else:
                    genpart_df = pd.read_parquet(genpart_outfile)
                
                if not os.path.exists(cl3d_outfile):
                    cl3d_df = grab_df(tree, cl3d_branches)
                    cl3d_df.to_parquet(cl3d_outfile)
                else:
                    cl3d_df = pd.read_parquet(cl3d_outfile)

                df_dict[key] = {'genpart': genpart_df, 'cl3d': cl3d_df}

    return df_dict

df_dict = generate_dfs(input_files)

def create_figures(phot_df, el_df, ptype, hist_bins):
    # Energy and PT Distributions
    fig_enpt = make_subplots(rows=1, cols=2, subplot_titles=(r"$\Huge{p_T}$", r"$\Huge{Energy}$"))

    fig_enpt.add_trace(go.Histogram(x=phot_df[f'good_{ptype}_pt'], nbinsx=hist_bins, marker_color=phot_color, autobinx=False, name='Photons', legendgroup='Photons'), row=1, col=1)
    fig_enpt.add_trace(go.Histogram(x=el_df[f'good_{ptype}_pt'], nbinsx=hist_bins, marker_color=el_color, autobinx=False, name='Electrons', legendgroup='Electrons'), row=1, col=1)

    fig_enpt.add_trace(go.Histogram(x=phot_df[f'good_{ptype}_energy'], nbinsx=hist_bins, marker_color=phot_color, autobinx=False, legendgroup='Photons', showlegend=False), row=1, col=2)
    fig_enpt.add_trace(go.Histogram(x=el_df[f'good_{ptype}_energy'], nbinsx=hist_bins, marker_color=el_color, autobinx=False, legendgroup='Electrons', showlegend=False), row=1, col=2)

    fig_enpt.update_layout(barmode='overlay',title_text=f'{ptype} Energy/PT Distributions', yaxis_title_text=r'$\Large{Events}$')
    fig_enpt.update_traces(opacity=0.5)

    # Eta and Phi Distributions
    fig_ang = make_subplots(rows=1, cols=2, subplot_titles=(r"$\Huge{\eta}$", r"$\Huge{\phi}$"))

    fig_ang.add_trace(go.Histogram(x=phot_df[f'good_{ptype}_eta'], nbinsx=hist_bins, marker_color=phot_color, autobinx=False, name='Photons', legendgroup='Photons'), row=1, col=1)
    fig_ang.add_trace(go.Histogram(x=el_df[f'good_{ptype}_eta'], nbinsx=hist_bins, marker_color=el_color, autobinx=False, name='Electrons', legendgroup='Electrons'), row=1, col=1)

    fig_ang.add_trace(go.Histogram(x=phot_df[f'good_{ptype}_phi'], nbinsx=hist_bins, marker_color=phot_color, autobinx=False, legendgroup='Photons', showlegend=False), row=1, col=2)
    fig_ang.add_trace(go.Histogram(x=el_df[f'good_{ptype}_phi'], nbinsx=hist_bins, marker_color=el_color, autobinx=False, legendgroup='Electrons', showlegend=False), row=1, col=2)

    fig_ang.update_layout(barmode='overlay', title_text=f'{ptype} Angular Distributions', yaxis_title_text=r'$\Large{Events}$')
    fig_ang.update_traces(opacity=0.5)

    return fig_enpt, fig_ang

#############################################################################################

dash.register_page(__name__, title='Pile Up', name='Pile Up')

layout = html.Div([
    html.H4('Some Pile Up Distributions'),

    html.Hr(),

    html.P("Binning:"),
    dcc.Slider(id='hist_bins', min=1, max=1000, step=100, value=400),

    html.P("EtaRange:"),
    dcc.RangeSlider(id='eta_range',min=1.4,max=2.7,step=0.1,value=[1.6,2.7]),

    html.P("Genpart/Reco:"),
    dcc.Dropdown(['Genpart', 'Reco'], 'Reco', id='particle_type'),

    html.P("PT Threshhold"),
    dcc.Slider(id='min_pt', min=0, max=10, step=1, value=0),

    html.P("Energy Threshhold"),
    dcc.Slider(id='min_en', min=0, max=10, step=1, value=0),

    html.Hr(),

    dcc.Graph(id='hist-distributions',mathjax=True),

    html.Hr(),
    dcc.Graph(id='angular-dists',mathjax=True)
])

#############################################################################################

@callback(
    Output("hist-distributions", "figure"),
    Output("angular-dists", "figure"),
    Input("hist_bins", "value"),
    Input("eta_range", "value"),
    Input("particle_type", "value"),
    Input("min_pt", "value"),
    Input("min_en", "value"))

def plot_dists(hist_bins, eta_range, particle_type, min_pt, min_en):

    # Cluster DFs
    phot_cl3d_df, el_cl3d_df = df_dict['photons']['cl3d'], df_dict['electrons']['cl3d']

    # Apply angular restrictions
    phot_cl3d_df = phot_cl3d_df[ (phot_cl3d_df.good_cl3d_eta > eta_range[0]) & (phot_cl3d_df.good_cl3d_eta < eta_range[1]) ]
    el_cl3d_df = el_cl3d_df[ (el_cl3d_df.good_cl3d_eta > eta_range[0]) & (el_cl3d_df.good_cl3d_eta < eta_range[1]) ]

    # Apply energy and pt thresholds
    phot_cl3d_df = phot_cl3d_df[ (phot_cl3d_df.good_cl3d_pt > min_pt) & (phot_cl3d_df.good_cl3d_energy > min_en) ]
    el_cl3d_df = el_cl3d_df[ (el_cl3d_df.good_cl3d_pt > min_pt) & (el_cl3d_df.good_cl3d_energy > min_en) ]

    # Genpart DFs
    phot_genpart_df, el_genpart_df = df_dict['photons']['genpart'], df_dict['electrons']['genpart']

    phot_genpart_df = phot_genpart_df[ (phot_genpart_df.good_genpart_eta > eta_range[0]) & (phot_genpart_df.good_genpart_eta < eta_range[1]) ]
    el_genpart_df = el_genpart_df[ (el_genpart_df.good_genpart_eta > eta_range[0]) & (el_genpart_df.good_genpart_eta < eta_range[1]) ]
    
    phot_genpart_df = phot_genpart_df[ (phot_genpart_df.good_genpart_pt > min_pt) & (phot_genpart_df.good_genpart_energy > min_en) ]
    el_genpart_df = el_genpart_df[ (el_genpart_df.good_genpart_pt > min_pt) & (el_genpart_df.good_genpart_energy > min_en) ]

    if particle_type=='Reco':
        fig_enpt, fig_ang = create_figures(phot_cl3d_df, el_cl3d_df, 'cl3d', hist_bins)
    elif particle_type=='Genpart':
        fig_enpt, fig_ang = create_figures(phot_genpart_df, el_genpart_df, 'genpart', hist_bins)
    else:
        print("\nSomething went wrong!")    

    return fig_enpt, fig_ang