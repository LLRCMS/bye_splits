import os
import sys
from threading import Timer
from dash import Dash, dcc, html, Input, Output, callback, dash_table
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
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

def effrms(data, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    out = {}
    x = np.sort(data, kind='mergesort')
    m = int(c *len(x)) + 1
    out = [np.min(x[m:] - x[:-m]) / 2.0]

    return out

phot_file = 'data/energy_out_ThresholdDummyHistomaxnoareath20_photon_0PU_truncation_hadd_with_pt_PAR_0p5_SEL_below_eta_2p7_REG_All_SW_1_SK_default_CA_min_distance.hdf5'
pion_file1 = 'data/energy_out_ThresholdDummyHistomaxnoareath20_ntuple_1_with_pt_PAR_0p5_SEL_below_eta_2p7_REG_All_SW_1_SK_default_CA_min_distance.hdf5'
pion_file2 = 'data/energy_out_ThresholdDummyHistomaxnoareath20_ntuple_2_with_pt_PAR_0p5_SEL_below_eta_2p7_REG_All_SW_1_SK_default_CA_min_distance.hdf5'
pion_file3 = 'data/energy_out_ThresholdDummyHistomaxnoareath20_ntuple_3_with_pt_PAR_0p5_SEL_below_eta_2p7_REG_All_SW_1_SK_default_CA_min_distance.hdf5'

'''def make_table(data):
    cols = data.keys()
    rows = [data[col] for col in cols]

    head = [html.Thead(html.Tr([html.Th(col) for col in cols]))]
    body = [html.Tbody(html.Tr([html.Td(row) for row in rows]))]

    return dbc.Table(head+body,bordered=True)'''

marks = {coef : {"label" : format(coef,'.3f'), "style": {"transform": "rotate(-90deg)"}} for coef in np.arange(0.0,0.05,0.001)}

dash.register_page(__name__, title='RMS', name='RMS')

layout = dbc.Container([
    dbc.Row([html.Div("Interactive Normal Distribution", style={'fontSize': 40, 'textAlign': 'center'})]),

    html.Hr(),

    dcc.Graph(id="histograms-x-graph",mathjax=True),
    html.P("Coef:"),
    dcc.Slider(id="coef", min=0.0, max=0.05, value=0.001,marks=marks),
    html.P("EtaRange:"),
    dcc.RangeSlider(id='eta_range',min=1.4,max=2.7,step=0.1,value=[1.4,2.7]),

    html.Hr(),

    dbc.Row([
        dcc.Markdown(r'Gaussianity := $\frac{|RMS-RMS_{Eff}|}{RMS}$', mathjax=True, style={'fontSize': 30, 'textAlign': 'center'})
    ]),

    html.Hr(),

    html.Div(id='my_table'),
])

@callback(
    Output("histograms-x-graph", "figure"),
    Output("my_table", "children"),
    Input("coef", "value"),
    Input("eta_range", "value"))

def display_color(coef, eta_range):
    with pd.HDFStore(phot_file,'r') as Phot, pd.HDFStore(pion_file1,'r') as Pi1, pd.HDFStore(pion_file2,'r') as Pi2, pd.HDFStore(pion_file3,'r') as Pi3:

        #coefs = np.linspace(0.0,0.05,50)[1:]
        coefs = np.linspace(0.0,0.05,50)

        coef_str = 'coef_{}'.format(str(coef).replace('.','p'))

        if coef_str not in Phot.keys():
            coef_list = [float(re.split('coef_',key)[1].replace('p','.')) for key in Phot.keys()]
            new_coef = closest(coef_list, coef)
            coef_str = 'coef_{}'.format(str(new_coef).replace('.','p'))

        phot_df = Phot[coef_str]
        pion_df = pd.concat([Pi1[coef_str],Pi2[coef_str],Pi3[coef_str]])

        phot_df['normed_energies'] = phot_df['en']/phot_df['genpart_energy']
        pion_df['normed_energies'] = pion_df['en']/pion_df['genpart_energy']

        phot_df = phot_df[ phot_df['genpart_exeta'] > eta_range[0] ]
        pion_df = pion_df[ pion_df['genpart_exeta'] > eta_range[0] ]

        phot_df = phot_df[ phot_df['genpart_exeta'] < eta_range[1] ]
        pion_df = pion_df[ pion_df['genpart_exeta'] < eta_range[1] ]

        phot_min_eta = phot_df['genpart_exeta'].min()
        phot_max_eta = phot_df['genpart_exeta'].max()

        pion_min_eta = pion_df['genpart_exeta'].min()
        pion_max_eta = pion_df['genpart_exeta'].max()

        phot_mean_en = phot_df['normed_energies'].mean()
        pion_mean_en = pion_df['normed_energies'].mean()

        phot_rms = phot_df['normed_energies'].std()/phot_mean_en
        phot_eff_rms = effrms(phot_df['normed_energies'])/phot_mean_en
        phot_gaus_diff = np.abs(phot_eff_rms-phot_rms)/phot_rms

        pion_rms = pion_df['normed_energies'].std()/pion_mean_en
        pion_eff_rms = effrms(pion_df['normed_energies'])/pion_mean_en
        pion_gaus_diff = np.abs(pion_eff_rms-pion_rms)/pion_rms

        pion_gaus_str = format(pion_gaus_diff[0], '.3f')
        phot_gaus_str = format(phot_gaus_diff[0], '.3f')

        pion_rms_str = format(pion_rms, '.3f')
        phot_rms_str = format(phot_rms, '.3f')

        pion_eff_rms_str = format(pion_eff_rms[0], '.3f')
        phot_eff_rms_str = format(phot_eff_rms[0], '.3f')

        fig = go.Figure()

        fig.add_trace(go.Histogram(x=pion_df['normed_energies'], nbinsx=100, autobinx=False, name='Pion'))
        fig.add_trace(go.Histogram(x=phot_df['normed_energies'], nbinsx=100, autobinx=False, name='Photon'))

        fig.update_layout(barmode='overlay',title_text='Normalized Cluster Energy', xaxis_title=r'$\Huge{\frac{E_{Cl}}{E_{Gen}}}$', yaxis_title_text=r'$\Large{Events}$')

        fig.update_traces(opacity=0.5)

        my_vals = {
            'Photon': {
                'RMS': phot_rms_str,
                'Effective RMS': phot_eff_rms_str,
                'Gaussianity': phot_gaus_str,
            },
            'Pion': {
                'RMS': pion_rms_str,
                'Effective RMS': pion_eff_rms_str,
                'Gaussianity': pion_gaus_str,
            }
        }

        val_df = pd.DataFrame(my_vals).reset_index()
        val_df = val_df.rename(columns={'index': ''})

        return fig, dbc.Table.from_dataframe(val_df)
