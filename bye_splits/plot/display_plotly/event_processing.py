_all_ = [ ]

import os
import pathlib
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

import data_handle
from data_handle.data_handle import EventDataParticle
from data_handle.geometry import GeometryData
from plot_event import produce_2dplot, produce_3dplot, plot_modules
from plotly.express.colors import sample_colorscale

import logging
log = logging.getLogger(__name__)

data_part_opt = dict(tag='v2', reprocess=False, debug=True, logger=log)
data_particle = {
    'photons 0PU': EventDataParticle(particles='photons', **data_part_opt),
    'photons 200PU': EventDataParticle(particles='photons_PU', **data_part_opt),
    'electrons 0PU': EventDataParticle(particles='electrons', **data_part_opt),
    'pions': EventDataParticle(particles='pions', **data_part_opt)}
geom_data = GeometryData(inname='/eos/user/m/mchiusi/visualization/test_triggergeom.root',
                         reprocess=False, logger=log)

axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white", showbackground=True, zerolinecolor="white",)

def sph2cart(eta, phi, z=322.):
    theta = 2*np.arctan(np.exp(-eta))
    r = z / np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    return x,y

def get_data(event, particles):
    ds_geom = geom_data.provide(library='plotly')
    
    if event is None:
        ds_ev, event = data_particle[particles].provide_random_event()
    else:
        ds_ev = data_particle[particles].provide_event(event)
     
    tc_keep = {'good_tc_waferu'     : 'waferu',
               'good_tc_waferv'     : 'waferv',
               'good_tc_cellu'      : 'triggercellu',
               'good_tc_cellv'      : 'triggercellv',
               'good_tc_layer'      : 'layer',
               'good_tc_energy'     : 'energy',
               'good_tc_pt'         : 'tc_pt',
               'good_tc_mipPt'      : 'mipPt',
               'good_tc_cluster_id' : 'tc_cluster_id'}
    
    gen_keep = {'good_genpart_exeta': 'exeta',
               'good_genpart_exphi' : 'exphi',
               'good_genpart_energy': 'gen_energy'} 
   
    gen_info = ds_ev['gen']
    gen_info = gen_info.rename(columns=gen_keep)
    
    ds_ev = ds_ev['tc']
    ds_ev = ds_ev.rename(columns=tc_keep)
    ds_ev = ds_ev[tc_keep.values()]
    
    ds_ev = pd.merge(left=ds_ev, right=ds_geom, how='inner',
                     on=['layer', 'waferu', 'waferv', 'triggercellu', 'triggercellv'])

    color = sample_colorscale('viridis', (ds_ev.mipPt-ds_ev.mipPt.min())/(ds_ev.mipPt.max()-ds_ev.mipPt.min()))
    ds_ev = ds_ev.assign(colors=color)
    return ds_ev, event, gen_info

def set_3dfigure(df):
    fig = go.Figure(produce_3dplot(df))
    
    fig.update_layout(autosize=False, width=1300, height=700,
                      scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1, y=1, z=1),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
                                 xaxis_title="z [cm]",yaxis_title="y [cm]",zaxis_title="x [cm]",
                                 xaxis_showspikes=False,yaxis_showspikes=False,zaxis_showspikes=False),
                      showlegend=False,
                      margin=dict(l=0, r=0, t=10, b=10),
                      ) 

    return fig

def update_3dfigure(fig, df):
    list_scatter = produce_3dplot(df, opacity=.2)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])
    return fig

def add_ROI(fig, df, k=4):
    ''' Choose k-layers window based on energy deposited in each layer '''
    layer_sums = df.groupby(['layer']).mipPt.sum() 
    initial_layer = (layer_sums.rolling(window=k).sum().shift(-k+1)).idxmax()
    
    mask = (df.layer>=initial_layer) & (df.layer<(initial_layer+2*k))
    input_df = df[mask]
    
    ''' Choose the (u,v) coordinates of the module corresponding to max dep-energy.
    This choice is performed by grouping modules beloging to different layers, having the same coordinates.
    Extend the selection to the nearby modules with at least 30% max energy and 10 mipT. '''
    module_sums = input_df.groupby(['waferu','waferv']).mipPt.sum()
    module_max = module_sums[module_sums.values == module_sums.max()].index[0]
    nearby_modules = [(module_max[0]+i, module_max[1]+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i*j >= 0]
    
    skim = (module_sums.index.isin(nearby_modules)) & (module_sums.values > 0.3 * module_sums.max()) & (module_sums.values > 10)
    skim_module_sums = module_sums[skim]
    
    roi_df = input_df[input_df.set_index(['waferu','waferv']).index.isin(skim_module_sums.index)]
    roi_df = roi_df.drop_duplicates(['waferu', 'waferv', 'layer'])

    list_scatter = plot_modules(roi_df)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])
    return fig

def set_2dfigure(df):
    fig = go.Figure(produce_2dplot(df))

    fig.update_layout(autosize=False, width=1300, height=700,
                      scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1, y=1),
                      scene=dict(xaxis=axis, yaxis=axis, 
                                 xaxis_title="x [cm]",yaxis_title="y [cm]",
                                 xaxis_showspikes=False,yaxis_showspikes=False),
                      showlegend=False,
                      margin=dict(l=0, r=0, t=10, b=10),
                      )

    return fig

def update_2dfigure(fig, df):
    list_scatter = produce_2dplot(df, opacity=.2)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])
    return fig


def layout(**options):
    return dbc.Container([html.Div([
        html.Div([
            html.Div([dcc.Dropdown(['photons 0PU', 'photons 200PU', 'electrons', 'pions'], 'photons 0PU', id='particle')], style={'width':'15%'}),
            html.Div([dbc.Checklist(options['checkbox'], [], inline=True, id='checkbox', switch=True)], style={"margin-left": "15px"}),
            html.Div(id='slider-container', children=html.Div(id='out_slider', style={'width':'95%'}), style= {'display': 'block', 'width':'40%'}),
        ], style={'display': 'flex', 'flex-direction': 'row'}),
    
        html.Div([
            html.Div(["Threshold in [mip\u209C]: ", dcc.Input(id='mip', value=1, type='number', step=0.1)], style={'padding': 10}),
            html.Div(["Select manually an event: ", dcc.Input(id='event', value=None, type='number')], style={'padding': 10, 'flex': 1}),
        ], style={'display':'flex', 'flex-direction':'row'}),
    
        html.Div([
            dbc.Button(children='Random event', id='event-val', n_clicks=0),
            dbc.Button(children='Submit selected event', id='submit-val', n_clicks=0, style={'display':'inline-block', "margin-left": "15px"}),
            html.Div(id='event-display', style={'display':'inline-block', "margin-left": "15px"}), 
        ], style={'display':'inline-block', "margin-left": "15px"}),

        dcc.Graph(id='plot'),
        dcc.Store(id='dataframe'),
        html.Div(id='page', key=options['page']),
        ]), ])
