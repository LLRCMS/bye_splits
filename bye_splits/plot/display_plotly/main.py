# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import argparse

import numpy as np
import pandas as pd
import event_processing as processing

app = Dash(__name__)
app.title = '3D Visualization' 
app.config['suppress_callback_exceptions'] = True

app.layout = html.Div([
    html.Div([
        html.Div([dcc.Dropdown(['3D view', 'Layer view'], '3D view', id='page')], style={'width':'15%'}),
    ], style={'display': 'flex', 'flex-direction': 'row'}),
    html.Div(id='page-content')
    ])

@app.callback(Output('page-content', 'children'),
              [Input('page', 'value')])
def render_content(page = '3D view'):
    if page == '3D view':
        return processing.tab_3d_layout
    elif page == 'Layer view':
        return processing.tab_layer_layout


@app.callback(Output('event-display', 'children'), Output('out_slider', 'children'), Output('dataframe', 'data'),
             [Input('particle', 'value'),  Input('tc-cl', 'value'),    Input('event-val', 'n_clicks'),
              Input('submit-val', 'n_clicks'), Input('mip', 'value'),  State('event', 'value')])
def update_event(particle, cluster, n_clicks, submit, mip, event):
    df, event  = processing.get_data(event, particle)

    slider = dcc.RangeSlider(df['layer'].min(),df['layer'].max(), value=[df['layer'].min(), df['layer'].max()], step=None,
                       marks={int(layer) : {"label": str(layer)} for each, layer in enumerate(sorted(df['layer'].unique()))}, 
                       id = 'slider-range')
    return u'Event {} selected'.format(event), slider, df.reset_index().to_json(date_format='iso')

@app.callback(Output('graph', 'figure'),  Output('slider-container', 'style'),
              [Input('submit-layer', 'n_clicks'), Input('dataframe', 'data')], 
              [Input('layer_sel', 'value'), State('tc-cl', 'value'), State('mip', 'value'), 
               State('slider-range', 'value'), State('page', 'value')])
def make_graph(submit, data, layer, cluster, mip, slider_value, page):
    df = pd.read_json(data, orient='records')
    df_sel = df[df.mipPt >= mip]
    
    if layer == 'layer selection':
        df_sel = df_sel[(df_sel.layer >= slider_value[0]) & (df_sel.layer <= slider_value[1])]
    
    if cluster == 'cluster':
        df_no_cluster = df_sel[df_sel.tc_cluster_id == 0]
        df_cluster    = df_sel[df_sel.tc_cluster_id != 0]
        fig = processing.set_3dfigure(df_cluster)
        fig = processing.update_3dfigure(fig, df_no_cluster)
    else:
        fig = processing.set_3dfigure(df_sel)
    
    if layer == 'display the entire event':
        status_slider = {'display': 'none', 'width':'1'}
    else: 
        status_slider = {'display': 'block', 'width':'1'}
    return fig, status_slider


@app.callback(Output('graph2d', 'figure'),
              [Input('dataframe', 'data'), Input('slider-range', 'value')],
              [State('tc-cl', 'value'), State('mip', 'value'), State('page', 'value')])
def make_graph(data, slider_value, cluster, mip, page):
    df = pd.read_json(data, orient='records')
    df_sel = df.loc[df.mipPt >= mip]

    df_sel = df_sel[df_sel.layer == slider_value[1]]
    if cluster == 'cluster':
        df_no_cluster = df_sel[df_sel.tc_cluster_id == 0]
        df_cluster    = df_sel[df_sel.tc_cluster_id != 0]
        fig = processing.set_2dfigure(df_cluster)
        fig = processing.update_2dfigure(fig, df_no_cluster)
    else:
        fig = processing.set_2dfigure(df_sel)
    return fig

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id','--username',type=str,default=os.getlogin())
    parser.add_argument('--host',type=str,default='llruicms01.in2p3.fr')
    parser.add_argument('--port',type=int,default=8004)
    args = parser.parse_args()
    
    app.run_server(debug=True,
                   host=args.host,
                   port=args.port)
