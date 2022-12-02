import os
import sys
from threading import Timer
from dash import Dash, dcc, html, Input, Output, callback
import dash
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

def effrms(data, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    out = {}
    x = np.sort(data, kind='mergesort')
    m = int(c *len(x)) + 1
    out = [np.min(x[m:] - x[:-m]) / 2.0]

    return out

def binned_effs(df, norm, perc=0.1):
    eff_list = [1.0]
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

layout = html.Div([
    html.H4('Reconstruction Efficiency'),
    dcc.Graph(id="eff-graph",mathjax=True),
    html.P("Coef:"),
    dcc.Slider(id="coef", min=0.0, max=0.05, value=0,marks=marks),
    html.P("EtaRange:"),
    dcc.RangeSlider(id='eta_range',min=1.4,max=2.7,step=0.1,value=[1.4,2.7]),
    dcc.Dropdown(['Energy', 'PT'], 'Energy', id='normby')
])

@callback(
    Output("eff-graph", "figure"),
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

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Photon", "Pion"))

        fig.add_trace(go.Scatter(x=phot_x, y=phot_effs, name='Photon'), row=1, col=1)
        fig.add_trace(go.Scatter(x=pion_x, y=pion_effs, name='Pion'), row=1, col=2)

        fig.update_xaxes(title_text='Energy (GeV)')

        fig.update_yaxes(type="log")

        fig.update_layout(title_text='Efficiency/Energy', yaxis_title_text=r'$Eff (\frac{N_{Cl}}{N_{Gen}})$')

        return fig
