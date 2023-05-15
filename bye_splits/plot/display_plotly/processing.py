_all_ = [ ]

import os
from pathlib import Path
import sys

parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from utils import params, common
from data_handle.data_process import *
from data_handle.geometry import GeometryData
from scripts.run_default_chain import run_chain_radii
from dash import dcc, html

import plot_event
import logging
log = logging.getLogger(__name__)

import h5py
import random

pars = {'no_fill': False, 'no_smooth': False, 'no_seed': False, 'no_cluster': False, 'no_validation': False, 'sel': 'all', 'reg': 'all', 'seed_window': 1, 'smooth_kernel': 'default', 'cluster_algo': 'min_distance', 'user': None, 'cluster_studies': False}

class Processing():
    def __init__(self):
        with open(params.CfgPath, "r") as afile:
            self.cfg = yaml.safe_load(afile)

        self.data_particle = { 'photons 0PU'  : 'photons',
                               'photons 200PU': 'photons_PU',
                               'electrons'    : 'electrons',
                               'pions'        : 'pions'}
        
        self.reprocess = self.cfg['3Ddisplay']['reprocess']
        self.coefs = self.cfg['3Ddisplay']['coefs']
        self.geom_data = GeometryData(reprocess=False, logger=log)
        self.ds_geom = self.geom_data.provide(library='plotly')
        self.filename = None
        self.list_events = None

    def random_event(self, f):
        hdf = h5py.File(f, 'r')
        if self.list_events == None:
            self.list_events = [ev[3:] for ev in list(hdf.keys())]
        return random.choice(list(hdf.keys()))[3:]

    def filestore(self, particle):
        self.filename = self.cfg["clusterStudies"]["localDir"] + self.cfg["3Ddisplay"][particle]

    def get_data(self, particle, event = ''): 
        self.filestore(self.data_particle[particle])      

        if not os.path.exists(self.filename) or self.reprocess:
             dict_ev, gen_info = run_chain_radii(common.dot_dict(pars), self.data_particle[particle], self.coefs)
             process_and_store(dict_ev, gen_info, self.ds_geom, self.filename)
  
        if event == None:
            event = self.random_event(self.filename)
        elif event not in self.list_events:
            dict_ev, gen_info = run_chain_radii(common.dot_dict(pars), self.data_particle[particle], self.coefs, event)
            process_and_store(dict_ev, gen_info, self.ds_geom, self.filename)
        dict_event = get_event(self.filename, event, self.coefs)
        
        return dict_event, event


    def layout(self, **options):
        return dbc.Container([html.Div([
            html.Div([
                html.Div([dcc.Dropdown(list(self.data_particle.keys()), 'pions', id='particle')], style={'width':'15%'}),
                html.Div([dbc.Checklist(options['checkbox'], [], inline=True, id='checkbox', switch=True)], style={"margin-left": "15px"}),
                html.Div(id='slider-container', children=html.Div(id='out_slider', style={'width':'99%'}), style= {'display': 'block', 'width':'55%'}),
            ], style={'display': 'flex', 'flex-direction': 'row'}),
        
            html.Div([
                html.Div(["Threshold in [mip\u209C]: ", dcc.Input(id='mip', value=1, type='number', step=0.1)], style={'padding': 10}),
                html.Div(["Select manually an event: ", dcc.Input(id='event', value=None, type='number')], style={'padding': 10, 'flex': 1}),
            ], style={'display':'flex', 'flex-direction':'row'}),
        
            html.Div([
                html.Div(["Select a particular clustering radius: "], style={'padding': 10}),
                html.Div([dcc.Slider(self.coefs[0], self.coefs[-1], 0.004,value=self.coefs[-1],id='slider_cluster')], style={'width':'60%'}),
            ], style={'display':'flex', 'flex-direction':'row'}),
    
            html.Div([
                dbc.Button(children='Random event', id='event-val', n_clicks=0),
                dbc.Button(children='Submit selected event', id='submit-val', n_clicks=0, style={'display':'inline-block', "margin-left": "15px"}),
                html.Div(id='event-display', style={'display':'inline-block', "margin-left": "15px"}), 
            ], style={'display':'inline-block', "margin-left": "15px"}),
    
            dcc.Graph(id='plot'),
            dcc.Store(id='dataframe'),
            dcc.Store(id='dataframe_sci'),
            html.Div(id='page', key=options['page']),
            ]), ])
