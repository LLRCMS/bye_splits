import os
import sys

from dash import dcc, html, Input, Output, callback
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import re
import pandas as pd

data_dir = '/eos/user/i/iehle/data/PU0/'

parent_dir = os.path.abspath(__file__ + 4 * '/..')
sys.path.insert(0, parent_dir)

import argparse
import bye_splits
from bye_splits.utils import params, parsing, common

parser = argparse.ArgumentParser(description='Clustering standalone step.')
parsing.add_parameters(parser)
FLAGS = parser.parse_args()
assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

FLAGS.reg = 'All'
FLAGS.sel = 'below_eta_2.7'

input_files = params.fill_kw['FillInFiles']

def_k = 0.0

# Find the element of a list containing strings ['coef_{float_1}', 'coef_{float_2}', ...] which is closest to some float_i
def closest(list, k=def_k):
    try:
        list = np.reshape(np.asarray(list), 1)
    except:
        list = np.asarray(list)
    if isinstance(k, str):
        k_num = float(re.split('coef_',k)[1].replace('p','.'))
    else:
        k_num = k
    id = (np.abs(list-k_num)).argmin()
    return list[id]

# Takes a dataframe 'df' with a column 'norm' to normalize by, and returns
# 1) a binned matching efficiency list
# 2) a binned list corresponding to 'norm' 
# where the binning is done by percentage 'perc' of the size of the 'norm' column
def binned_effs(df, norm, perc=0.1):
    eff_list = [0]
    en_list = [0]
    en_bin_size = perc*(df[norm].max() - df[norm].min())
    if not perc==1.:
        print("This would be very weird...")
        current_en = 0
        for i in range(100):
            match_column = df.loc[df[norm].between(current_en, (i+1)*en_bin_size, 'left'), 'matches']
            if not match_column.empty:
                try:
                    eff = float(match_column.value_counts(normalize=True))
                except:
                    eff = match_column.value_counts(normalize=True)[True]
                eff_list.append(eff)
                current_en += en_bin_size
                en_list.append(current_en)
    else:
        match_column = df.loc[df[norm].between(0, en_bin_size, 'left'), 'matches']
        if norm != "Energy":
            print("\nEff Troubleshooting\n======================\n")
            print("\nNormalization: {}".format(norm))
            print("\nBin size: {}".format(en_bin_size))
            print("\nDF: \n{}\n".format(df))
            print("\nMatches: \n{}\n".format(match_column))
        if not match_column.empty:
            try:
                eff = float(match_column.value_counts(normalize=True))
            except:
                eff = match_column.value_counts(normalize=True)[True]
            eff_list = eff
        print("\nEff: {}".format(eff_list))
    return eff_list, en_list

def get_str(coef, file):
    coef_str = 'coef_{}'.format(str(coef).replace('.','p'))
    if coef_str not in file.keys():
        coef_list = [float(re.split('coef_',key)[1].replace('p','.')) for key in file.keys()]
        new_coef = closest(coef_list, coef)
        coef_str = 'coef_{}'.format(str(new_coef).replace('.','p'))
    return coef_str

# Goal: Write function that finds and returns the DataFrames corresponding to a chosen coefficient in these files without explicitly referencing them
def get_dfs(init_files, coef, pars):

    file_dict = {}
    start = params.energy_kw['EnergyOut']
    df_dict = dict.fromkeys(init_files.keys(),[0.0])

    for key in init_files.keys():
        file_dict[key] = [start+re.split('gen_cl3d_tc',file)[1] for file in init_files[key]]
        file_dict[key] = [common.fill_path(file,**pars) for file in file_dict[key]]
        if len(file_dict[key])==1:

            File = pd.HDFStore(file_dict[key][0],'r')
            
            if not isinstance(coef, str):
                coef = get_str(coef, File)
            
            df = File[coef]
        else:
            file_list = [pd.HDFStore(val,'r') for val in file_dict[key]]
            if not isinstance(coef, str):
                coef = get_str(coef, file_list[0])
            df_list = [file_list[i][coef] for i in range(len(file_list))]
            df = pd.concat(df_list)
        df_dict[key] = df
    
    return df_dict

def get_keys(init_files, pars):

    start = params.energy_kw['EnergyOut']

    file_name = start+re.split('gen_cl3d_tc',init_files['photon'][0])[1]
    file_path = common.fill_path(file_name, **pars)
    
    # testing
    print("\nFilepath: {}\n".format(file_path))

    with pd.HDFStore(file_path, 'r') as File:
        keys = File.keys()

    return keys

# Dash page setup
##############################################################################################################################

marks = {coef : {"label" : format(coef,'.3f'), "style": {"transform": "rotate(-90deg)"}} for coef in np.arange(0.0,0.05,0.001)}

dash.register_page(__name__, title='Efficiency', name='Efficiency')

layout = dbc.Container([
    dbc.Row([html.Div('Reconstruction Efficiency', style={'fontSize': 30, 'textAlign': 'center'})]),

    html.Hr(),

    dcc.Graph(id="eff-graph",mathjax=True),

    html.P("Coef:"),
    dcc.Slider(id="coef", min=0.0, max=0.05, value=0.001,marks=marks),

    html.P("EtaRange:"),
    dcc.RangeSlider(id='eta_range',min=1.4,max=2.7,step=0.1,value=[1.6,2.7]),

    html.P("Normalization:"),
    dcc.Dropdown(['Energy', 'PT'], 'PT', id='normby'),

    html.Hr(),

    dbc.Row([
        dcc.Markdown("Global Efficiencies", style={'fontSize': 30, 'textAlign': 'center'})
    ]),

    html.Div(id='glob-effs'),

    html.Hr(),

    dbc.Row([
        dcc.Markdown("Efficiencies By Coefficent", style={'fontSize': 30, 'textAlign': 'center'})
    ]),

    dcc.Graph(id='glob-eff-graph', mathjax=True)

])

# Callback function for display_color() which displays binned efficiency/energy graphs
@callback(
    Output("eff-graph", "figure"),
    Output("glob-effs", "children"),
    Input("coef", "value"),
    Input("eta_range", "value"),
    Input("normby", "value"))

##############################################################################################################################

def display_color(coef, eta_range, normby):
    df_by_particle = get_dfs(input_files, coef, pars=vars(FLAGS))
    phot_df = df_by_particle['photon']
    pion_df = df_by_particle['pion']

    phot_df = phot_df[ (phot_df['genpart_exeta'] > eta_range[0]) ]
    pion_df = pion_df[ (pion_df['genpart_exeta'] > eta_range[0]) ]
    phot_df = phot_df[ (phot_df['genpart_exeta'] < eta_range[1]) ]
    pion_df = pion_df[ (pion_df['genpart_exeta'] < eta_range[1]) ]

    # Bin energy data into n% chunks to check eff/energy (10% is the default)
    if normby=='Energy':
        phot_effs, phot_x = binned_effs(phot_df, 'genpart_energy')
        pion_effs, pion_x = binned_effs(pion_df, 'genpart_energy')
    else:
        phot_effs, phot_x = binned_effs(phot_df, 'genpart_pt')
        pion_effs, pion_x = binned_effs(pion_df, 'genpart_pt')

    glob_effs = pd.DataFrame({'Photon': np.mean(phot_effs[1:]),
                    'Pion': np.mean(pion_effs[1:])
                    }, index=[0])

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

    fig.add_trace(go.Scatter(x=phot_x, y=phot_effs, name='Photon'), row=1, col=1)
    fig.add_trace(go.Scatter(x=pion_x, y=pion_effs, name='Pion'), row=1, col=2)

    fig.update_xaxes(title_text="{} (GeV)".format(normby))

    fig.update_yaxes(type="log")

    fig.update_layout(title_text="Efficiency/{}".format(normby), yaxis_title_text=r'$Eff (\frac{N_{Cl}}{N_{Gen}})$')

    return fig, dbc.Table.from_dataframe(glob_effs)

def write_eff_file(norm, coefs, eta, file):    
    effs_dict = {'Photon': [0.0],
                        'Pion': [0.0]}
    for coef in coefs[1:]:
        dfs_by_particle = get_dfs(input_files, coef, pars=vars(FLAGS))
        phot_df = dfs_by_particle['photon']
        pion_df = dfs_by_particle['pion']

        phot_df = phot_df[ (phot_df['genpart_exeta'] > eta[0]) ]
        pion_df = pion_df[ (pion_df['genpart_exeta'] > eta[0]) ]
        phot_df = phot_df[ (phot_df['genpart_exeta'] < eta[1]) ]
        pion_df = pion_df[ (pion_df['genpart_exeta'] < eta[1]) ]

        if norm=='Energy':
            phot_eff, _ = binned_effs(phot_df, 'genpart_energy', 1.)
            pion_eff, _ = binned_effs(pion_df, 'genpart_energy', 1.)
        else:
            phot_eff, _ = binned_effs(phot_df, 'genpart_pt', 1.)
            pion_eff, _ = binned_effs(pion_df, 'genpart_pt', 1.)

        effs_dict['Photon'] = np.append(effs_dict['Photon'], phot_eff)
        effs_dict['Pion'] = np.append(effs_dict['Pion'], pion_eff)
    
    with pd.HDFStore(file, "w") as glob_eff_file:
        glob_eff_file.put('Eff', pd.DataFrame.from_dict(effs_dict))
    
    return effs_dict

# Callback function for global_effs() which displays global efficiency as a function of the coefficent/radius
@callback(
    Output("glob-eff-graph", "figure"),
    Input("eta_range", "value"),
    Input("normby", "value")
)

def global_effs(eta_range, normby, file="global_eff.hdf5"):

    coefs = get_keys(input_files, pars=vars(FLAGS))

    filename = "{}{}_eta_{}_{}_{}".format(data_dir, normby, eta_range[0], eta_range[1], file)

    if not os.path.exists(filename):    
        effs_by_coef = write_eff_file(normby, coefs, eta_range, filename)
    
    else:
        try:
            with pd.HDFStore(filename, "r") as glob_eff_file:
                effs_by_coef = glob_eff_file['/Eff'].to_dict(orient='list')
        except:
            os.remove(filename)
            effs_by_coef = write_eff_file(normby, coefs, eta_range, filename)

    print("\nGlobal Eff Troubleshooting: \n===========================\n")
    print("\nEff Dict: \n{}\n".format(effs_by_coef))

    coefs = np.linspace(0.0,0.05,50)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

    fig.add_trace(go.Scatter(x=coefs, y=effs_by_coef['Photon'], name='Photon'), row=1, col=1)
    fig.add_trace(go.Scatter(x=coefs, y=effs_by_coef['Pion'], name='Pion'), row=1, col=2)

    fig.update_xaxes(title_text='Radius (Coefficient)')

    # Range [a,b] is defined by [10^a, 10^b], hence passing to log
    fig.update_yaxes(type='log', range=[np.log10(0.997), np.log(1.001)])

    fig.update_layout(title_text='Efficiency/Radius', yaxis_title_text=r'$Eff (\frac{N_{Cl}}{N_{Gen}})$')

    return fig
