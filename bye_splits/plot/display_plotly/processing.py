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

pars = {'no_fill': False, 'no_smooth': False, 'no_seed': False, 'no_cluster': False, 'no_validation': False, 'sel': 'all', 'reg': 'Si', 'seed_window': 1, 'smooth_kernel': 'default', 'cluster_algo': 'min_distance', 'user': None, 'cluster_studies': False}

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
        self.geom_data = GeometryData(reprocess=self.reprocess, logger=log)
        self.ds_geom = self.geom_data.provide(library='plotly')
        self.filename = None

    def random_event(self, f):
        hdf = h5py.File(f, 'r')
        return random.choice(list(hdf.keys()))[3:]

    def filestore(self, particle):
        self.filename = self.cfg["clusterStudies"]["localDir"] + self.cfg["3Ddisplay"][particle]

    def get_data(self, event, particle): 
        self.filestore(self.data_particle[particle])
        
        if not os.path.exists(self.filename) or self.reprocess:
             dict_ev, gen_info = run_chain_radii(common.dot_dict(pars), self.data_particle[particle], self.coefs)
             process_and_store(dict_ev, gen_info, self.ds_geom, self.filename)
    
        event = self.random_event(self.filename)
        dict_event = get_event(self.filename, event, self.coefs)
   

        print(self.ds_geom['si'].columns)     
        print(self.ds_geom['sci'].columns)     
        # if particles == 'photons 200PU':
        #     for coef in coefs:    
        #         eta = gen_info['gen_eta'].values[0]
        #         phi = gen_info['gen_phi'].values[0]
        #         x_gen, y_gen = sph2cart(eta, phi)
        #         ds_ev[coef] = ds_ev[coef][np.sqrt((x_gen-ds_ev[coef].tc_x)**2+(y_gen-ds_ev[coef].tc_y)**2)<50]
        return dict_event, event


    def roi_finder(self, input_df, threshold=30, nearby=False):
        ''' Choose the (u,v) coordinates of the module corresponding to max dep-energy.
        This choice is performed by grouping modules beloging to different layers, having the same coordinates.
        Extend the selection to the nearby modules with at least 30% max energy and 10 mipT. '''
        module_sums = input_df.groupby(['waferu','waferv']).mipPt.sum()
        module_ROI = list(module_sums[module_sums.values >= threshold].index)
    
        if nearby:
            selected_modules = []
            for module in module_ROI:
                nearby_modules = [(module[0]+i, module[1]+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i*j >= 0]
                
                skim = (module_sums.index.isin(nearby_modules)) & (module_sums.values > 0.3 * module_sums[module]) & (module_sums.values > 10)
                skim_nearby_module = list(module_sums[skim].index)
                selected_modules.extend(skim_nearby_module)
            module_ROI = selected_modules
        
        roi_df = input_df[input_df.set_index(['waferu','waferv']).index.isin(module_ROI)]
        return roi_df.drop_duplicates(['waferu', 'waferv', 'layer']), module_ROI
    
    
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
