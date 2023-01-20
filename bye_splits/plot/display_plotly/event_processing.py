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

import data_handle
from data_handle.data_handle import EventDataParticle
from data_handle.geometry import GeometryData
from plot_event import produce_2dplot, produce_3dplot
from plotly.express.colors import sample_colorscale

import logging
log = logging.getLogger(__name__)

data_part_opt = dict(tag='v2', reprocess=False, debug=True, logger=log)
data_particle = {
    'photons': EventDataParticle(particles='photons', **data_part_opt),
    'electrons': EventDataParticle(particles='electrons', **data_part_opt),
    'pions': EventDataParticle(particles='pions', **data_part_opt)}
geom_data = GeometryData(inname='test_triggergeom.root',
                         reprocess=False, logger=log)

axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white", showbackground=True, zerolinecolor="white",)

def get_data(event, particles):
    ds_geom = geom_data.provide(library='plotly')
   
    if event is None:
    	event = data_particle[particles].provide_event_numbers()
    
    ds_ev = data_particle[particles].provide_event(event)
   
    ds_ev.rename(columns={'good_tc_waferu':'waferu', 'good_tc_waferv':'waferv',
                          'good_tc_cellu':'triggercellu', 'good_tc_cellv':'triggercellv',
                          'good_tc_layer':'layer', 'good_tc_mipPt':'mipPt',
                          'good_tc_cluster_id':'tc_cluster_id'},
                inplace=True)
    ds_ev = pd.merge(left=ds_ev, right=ds_geom, how='inner',
                     on=['layer', 'waferu', 'waferv', 'triggercellu', 'triggercellv'])

    color = sample_colorscale('viridis', (ds_ev.mipPt-ds_ev.mipPt.min())/(ds_ev.mipPt.max()-ds_ev.mipPt.min()))
    ds_ev = ds_ev.assign(colors=color)
    return ds_ev, event

def set_3dfigure(df):
    fig = go.Figure(produce_3dplot(df))
    
    fig.update_layout(autosize=False, width=1800, height=850,
                      scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1, y=1, z=1),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
                                 xaxis_title="x [cm]",yaxis_title="y [cm]",zaxis_title="z [cm]"),
                      ) 

    return fig

def update_3dfigure(fig, df):
    list_scatter = produce_3dplot(df, opacity=.2)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])
    return fig

def set_2dfigure(df):
    fig = go.Figure(produce_2dplot(df))

    fig.update_layout(autosize=False, width=1500, height=850,
                      scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1, y=1),
                      scene=dict(xaxis=axis, yaxis=axis, 
                                 xaxis_title="x [cm]",yaxis_title="y [cm]"),
                      )

    return fig

def update_2dfigure(fig, df):
    list_scatter = produce_2dplot(df, opacity=.2)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])
    return fig


tab_3d_layout = html.Div([
    html.H3('3D TC distribution'),
    html.Div([
        html.Div([dcc.Dropdown(['photons', 'electrons', 'pions'], 'photons', id='particle')], style={'width':'15%'}),
        html.Div([dcc.Dropdown(['trigger cells', 'cluster'], 'trigger cells', id='tc-cl')], style={"margin-left": "15px", 'width':'15%'}),
        html.Div(id='space-slider', style={'width':'15%'}),
        html.Div([dcc.Dropdown(['display the entire event', 'layer selection'], 'layer selection', id='layer_sel')], style={"margin-left": "15px", 'width':'15%'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div([
        html.Div(["Threshold in [MIP Pt]: ", dcc.Input(id='mip', value=1, type='number', step=0.1)], style={'padding': 10}),
        html.Div(["Select manually an event: ", dcc.Input(id='event', value=None, type='number')], style={'padding': 10, 'flex': 1}),
        html.Div(id='slider-container', children=html.Div(id='out_slider', style={'width':'95%'}), style= {'display': 'block', 'width':'30%'}),
        html.Div(id='space-slider', style={'width':'30%'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Br(),
    html.Div([
        html.Button(children='Random event', id='event-val', n_clicks=0),
        html.Button(children='Submit selected event', id='submit-val', n_clicks=0),
        html.Button(children='Submit selected layer(s)', id='submit-layer', n_clicks=0),
        html.Div(id='event-display', style={'display':'inline-block', "margin-left": "15px"}),
        html.Div(id='which', style={'display':'inline-block', "margin-left": "15px"}),
    ]),
    dcc.Graph(id='graph'),
    dcc.Store(id='dataframe'),
    ])

tab_layer_layout = html.Div([
    html.H3('2D TC distribution - Layer view'),
    html.Div([
        html.Div([dcc.Dropdown(['photons', 'electrons', 'pions'], 'photons', id='particle')], style={'width':'15%'}),
        html.Div([dcc.Dropdown(['trigger cells', 'cluster'], 'trigger cells', id='tc-cl')], style={"margin-left": "15px", 'width':'15%'}),
        html.Br(),
        html.Div(id='out_slider', style={'width':'30%'}),
        html.Div(id='layer_slider_container', style={'width':'30%'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div([
        html.Div(["Threshold in [MIP Pt]: ", dcc.Input(id='mip', value=1, type='number', step=0.1)], style={'padding': 10}),
        html.Div(["Select manually an event: ", dcc.Input(id='event', value=None, type='number')], style={'padding': 10, 'flex': 1}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Br(),
    html.Div([
        html.Button(children='Random event', id='event-val', n_clicks=0),
        html.Button(children='Submit selected event', id='submit-val', n_clicks=0),
        html.Div(id='event-display', style={'display':'inline-block', "margin-left": "15px"}),
        html.Div(id='which', style={'display':'inline-block', "margin-left": "15px"}),
    ]),
    dcc.Graph(id='graph2d'),
    dcc.Store(id='dataframe')])
