_all_ = [ ]

import os
from pathlib import Path
import sys

parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import yaml
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc

from dash import dcc, html
import dash_daq as daq
import h5py
import re
import plotly.express.colors as px
import random
import logging
log = logging.getLogger(__name__)

from utils import params, common 
from data_handle.data_process import *
from data_handle.geometry import GeometryData
from scripts.run_radii_chain import run_radii_chain

class Processing():
    def __init__(self):
        with open(params.CfgPath, "r") as afile:
            self.cfg = yaml.safe_load(afile)

        self.reprocess = self.cfg['3Ddisplay']['reprocess']
        self.coefs = self.cfg['3Ddisplay']['coefs']
        self.chain = {'default' : 'Default chain',
                      'cs'      : 'Coarse seeding chain'}
        self.ds_geom = GeometryData(reprocess=False, logger=log, library='plotly').provide()
        self.filename = None
        self.list_events = None

    def random_event(self, f):
        return random.choice(self.list_events)

    def filestore(self, particles, pu):
        self.filename = self.cfg["clusterStudies"]["localDir"] + self.cfg["3Ddisplay"][f"PU{pu}"][particles] + ".hdf5"

    def get_data(self, pars, particles, pu_str, event = ''):
        pu = re.search(r'(\d+) pileup', pu_str).group(1)
        self.filestore(particles, pu)      

        if not os.path.exists(self.filename) or self.reprocess:
            dict_ev, gen_info = run_radii_chain(pars, particles, pu, self.coefs)
            self.process_event(dict_ev, gen_info)
            
        with h5py.File(self.filename, 'r') as file: 
            self.list_events = [key.split('ev_')[1] for key in file.keys()]

        event = event or self.random_event(self.filename)
        if str(event) not in self.list_events:
            dict_ev, gen_info = run_radii_chain(pars, particles, pu, self.coefs, event)
            self.process_event(dict_ev, gen_info)
        
        dict_event = self.get_event(event)
        return dict_event, event

    def get_event(self, event):
        """   Load dataframes of a particular event from the hdf5 file   """
        dic = {}
        for chain in self.chain:
          dic[chain] = {}
          for coef in self.coefs:
            key = '/ev_'+str(event)+'/'+chain+'/coef_'+str(coef)[2:] # deleting 0. in the coef strings
            dic[chain][coef] = pd.read_hdf(self.filename, key)
        dic['gen'] = pd.read_hdf(self.filename, '/ev_'+str(event)+'/gen_info')
        return dic
    
    def store_event(self, path, data):
        """   Save dictionary in a h5file. A dictionary corresponds to one event.
        The key is a selected radius, while the values is the corresponding dataframe  """
    
        if isinstance(data, dict):
            for key, item in data.items():
                if isinstance(item, pd.DataFrame):
                    item.to_hdf(self.filename, path + str(key))
                else:
                    raise ValueError('Cannot save %s type'%type(item))
        else:
            data.to_hdf(self.filename, path)
    
    def process_event(self, dict_events, gen_info):
    
        tc_keep = {'seed_idx'     : 'seed_idx',
                   'mipPt'        : 'tc_mipPt',
                   'tc_eta'       : 'tc_eta',
                   'waferu'       : 'tc_wu',
                   'waferv'       : 'tc_wv',
                   'triggercellu' : 'tc_cu',
                   'triggercellv' : 'tc_cv',
                   'layer'        : 'tc_layer'}
    
        sci_update = {'triggercellieta' : 'tc_cu',
                      'triggercelliphi' : 'tc_cv',
                      'layer'           : 'tc_layer',
                      'waferu'          : 'tc_wu'}
        
        self.ds_geom['si']  = self.ds_geom['si'].rename(columns=tc_keep)
        self.ds_geom['sci'] = self.ds_geom['sci'].rename(columns=sci_update)
       
        for event in dict_events.keys():
          print('Procesing event '+str(event))
          for chain in dict_events[event].keys():
            dict_event = dict_events[event][chain]
            for coef in dict_event.keys():
                silicon_df = pd.merge(left=dict_event[coef], right=self.ds_geom['si'], how='inner',
                                      on=['tc_layer', 'tc_wu', 'tc_wv', 'tc_cu', 'tc_cv'])
                silicon_df = silicon_df.drop(['waferorient', 'waferpart'], axis=1)
                scintillator_df = pd.merge(left=dict_event[coef], right=self.ds_geom['sci'], how='inner',
                                           on=['tc_layer', 'tc_wu', 'tc_cu', 'tc_cv'])
    
                dict_event[coef] = pd.concat([silicon_df, scintillator_df]).reset_index(drop=True)
                color_continuous = common.colorscale(dict_event[coef], 'tc_mipPt', 'viridis')
                color_discrete   = common.colorscale(dict_event[coef], 'seed_idx', px.qualitative.Light24, True)
                dict_event[coef] = dict_event[coef].assign(color_energy=color_continuous, color_clusters=color_discrete)
                
            self.store_event('/ev_'+str(event)+'/'+chain+'/coef_', dict_event)
            self.store_event('/ev_'+str(event)+'/gen_info', gen_info[gen_info.event == event])


    def layout(self, **options):
        return dbc.Container([html.Div([
            html.Div([
                html.Div([dcc.Dropdown(['photons', 'electrons', 'pions'], 'photons', id='particle')], style={'width':'12%'}),
                html.Div([dcc.Dropdown(['0 pileup', '200 pileup'], '0 pileup', id='pu')], style={'width':'12%', "margin-left": "10px"}),
                html.Div([dbc.Checklist(options, [], inline=True, id='checkbox', switch=True)], style={"margin-left": "15px"}),
                html.Div(id='slider-container', children=html.Div(id='out_slider', style={'width':'99%'}), style= {'display': 'block', 'width':'55%'}),
            ], style={'display': 'flex', 'flex-direction': 'row'}),

            html.Div([
		html.Div([dcc.Markdown('Threshold in mip$_{\\text{T}}$: ', mathjax=True, style={'margin-left': '17px','margin-bottom': '-10px'}), dcc.Input(id='mip', value=1, type='number', step=0.1)], style={'padding': '5px', 'margin-bottom': '0px', 'margin-right': '0px', 'width': '23%'}),
                html.Div([dcc.Markdown("Select manually an event: ", style={'margin-left': '5px','margin-bottom': '-10px'}), dcc.Input(id='event', value=None, type='number')], style={'padding': '5px', 'margin-right': '20px', 'margin-bottom': '-2px', 'width': '23%'}),
                html.Div([dcc.Markdown("Select a reconstruction chain: ", style={'margin-left': '15px','margin-bottom': '-15px'}), dcc.Dropdown(id='chain', options=self.chain, value='default')], style={'padding': '5px', 'margin-right': '5px', 'margin-bottom': '0px', 'width': '24%'}),
                html.Div([dcc.Markdown("Select the clustering radius: ", style={'margin-left': '25px','margin-bottom': '-10px'}), dcc.Slider(self.coefs[0], self.coefs[-1], 0.005, value=self.coefs[-1], id='slider_cluster', included=False)], style={'padding': '5px', 'margin-left': '20px', 'width': '30%'}),
            ], style={'display':'flex','flex-direction':'row', 'marginRight': 250, 'marginLeft': 0}),

            html.Div([
                dbc.Button(children='Random event', id='event-val', n_clicks=0),
                dbc.Button(children='Submit selected event', id='submit-val', n_clicks=0, style={'display':'inline-block', "margin-left": "15px"}),
                html.Div(children=[dcc.Markdown(mathjax=True, id='event-display')], style={'display':'inline-block', "margin-left": "15px"}), 
            ], style={'display':'inline-block', "margin-left": "15px"}),
    
            dcc.Loading(id="loading", children=[html.Div(dcc.Graph(id='plot'))], type="circle"),
            dcc.Store(id='dataframe'),
            dcc.Store(id='dataframe_sci'),
            html.Div(id='page', key=options['page']),
            ]), ])
