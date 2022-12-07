import os
import sys
from threading import Timer
from dash import Dash, dcc, html, Input, Output, callback
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import pandas as pd
import webbrowser

parent_dir = os.path.abspath(__file__ + 4 * '/..')
sys.path.insert(0, parent_dir)

from bye_splits.utils import params, common

def_k = 0.0

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

def binned_effs(df, norm, perc=0.1):
    eff_list = [0]
    en_list = [0]
    en_bin_size = perc*(df[norm].max() - df[norm].min())
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
    return eff_list, en_list

phot_file = 'data/energy_out_ThresholdDummyHistomaxnoareath20_photon_0PU_truncation_hadd_with_pt_PAR_0p5_SEL_below_eta_2p7_REG_All_SW_1_SK_default_CA_min_distance.hdf5'
pion_file1 = 'data/energy_out_ThresholdDummyHistomaxnoareath20_ntuple_1_with_pt_PAR_0p5_SEL_below_eta_2p7_REG_All_SW_1_SK_default_CA_min_distance.hdf5'
pion_file2 = 'data/energy_out_ThresholdDummyHistomaxnoareath20_ntuple_2_with_pt_PAR_0p5_SEL_below_eta_2p7_REG_All_SW_1_SK_default_CA_min_distance.hdf5'
pion_file3 = 'data/energy_out_ThresholdDummyHistomaxnoareath20_ntuple_3_with_pt_PAR_0p5_SEL_below_eta_2p7_REG_All_SW_1_SK_default_CA_min_distance.hdf5'

marks = {coef : {"label" : format(coef,'.3f'), "style": {"transform": "rotate(-90deg)"}} for coef in np.arange(0.0,0.05,0.001)}

dash.register_page(__name__, title='Efficiency', name='Efficiency')

layout = dbc.Container([
    dbc.Row([html.Div('Reconstruction Efficiency', style={'fontSize': 30, 'textAlign': 'center'})]),

    html.Hr(),

    dcc.Graph(id="eff-graph",mathjax=True),

    html.P("Coef:"),
    dcc.Slider(id="coef", min=0.0, max=0.05, value=0,marks=marks),

    html.P("EtaRange:"),
    dcc.RangeSlider(id='eta_range',min=1.4,max=2.7,step=0.1,value=[1.4,2.7]),

    html.P("Normalization:"),
    dcc.Dropdown(['Energy', 'PT'], 'Energy', id='normby'),

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

@callback(
    Output("eff-graph", "figure"),
    Output("glob-effs", "children"),
    Input("coef", "value"),
    Input("eta_range", "value"),
    Input("normby", "value"))

def display_color(coef, eta_range, normby):
    with pd.HDFStore(phot_file,'r') as Phot, pd.HDFStore(pion_file1,'r') as Pi1, pd.HDFStore(pion_file2,'r') as Pi2, pd.HDFStore(pion_file3,'r') as Pi3:

        coefs = np.linspace(0.0,0.05,50)

        coef_str = 'coef_{}'.format(str(coef).replace('.','p'))

        if coef_str not in Phot.keys():
            coef_list = [float(re.split('coef_',key)[1].replace('p','.')) for key in Phot.keys()]
            new_coef = closest(coef_list, coef)
            coef_str = 'coef_{}'.format(str(new_coef).replace('.','p'))

        phot_df = Phot[coef_str]
        pion_df = pd.concat([Pi1[coef_str],Pi2[coef_str],Pi3[coef_str]])

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

        fig.update_xaxes(title_text='Energy (GeV)')

        fig.update_yaxes(type="log")

        fig.update_layout(title_text='Efficiency/Energy', yaxis_title_text=r'$Eff (\frac{N_{Cl}}{N_{Gen}})$')

        return fig, dbc.Table.from_dataframe(glob_effs)

@callback(
    Output("glob-eff-graph", "figure"),
    Input("eta_range", "value"),
    Input("normby", "value")
)

def global_effs(eta_range, normby):
    with pd.HDFStore(phot_file,'r') as Phot, pd.HDFStore(pion_file1,'r') as Pi1, pd.HDFStore(pion_file2,'r') as Pi2, pd.HDFStore(pion_file3,'r') as Pi3:

        effs_by_coef = {'Photon': [0.0],
                        'Pion': [0.0]}

        for coef in Phot.keys()[1:]:
            phot_df = Phot[coef]
            pion_df = pd.concat([Pi1[coef],Pi2[coef],Pi3[coef]])

            phot_df = phot_df[ (phot_df['genpart_exeta'] > eta_range[0]) ]
            pion_df = pion_df[ (pion_df['genpart_exeta'] > eta_range[0]) ]
            phot_df = phot_df[ (phot_df['genpart_exeta'] < eta_range[1]) ]
            pion_df = pion_df[ (pion_df['genpart_exeta'] < eta_range[1]) ]

            phot_eff = phot_df['matches'].value_counts(normalize=True)
            pion_eff = pion_df['matches'].value_counts(normalize=True)

            try:
                phot_eff = phot_eff[True]
                pion_eff = pion_eff[True]
            except:
                print("Troubleshooting...")
                quit()

            effs_by_coef['Photon'] = np.append(effs_by_coef['Photon'], phot_eff)
            effs_by_coef['Pion'] = np.append(effs_by_coef['Pion'], pion_eff)

        coefs = np.linspace(0.0,0.05,50)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

        fig.add_trace(go.Scatter(x=coefs, y=effs_by_coef['Photon'], name='Photon'), row=1, col=1)
        fig.add_trace(go.Scatter(x=coefs, y=effs_by_coef['Pion'], name='Pion'), row=1, col=2)

        fig.update_xaxes(title_text='Radius (Coefficient)')

        # Range [a,b] is defined by [10^a, 10^b], hence passing to log
        fig.update_yaxes(type='log', range=[np.log10(0.997), np.log(1.001)])

        fig.update_layout(title_text='Efficiency/Radius', yaxis_title_text=r'$Eff (\frac{N_{Cl}}{N_{Gen}})$')

        return fig

#global_effs([1.4,2.7],normby='Energy')
