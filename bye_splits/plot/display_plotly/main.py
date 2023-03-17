# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import argparse
import numpy as np
import pandas as pd
import event_processing as processing

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = '3D Visualization' 
app.config['suppress_callback_exceptions'] = True
load_figure_template('FLATLY')

app.layout = html.Div([
    html.Div([dbc.NavbarSimple([
                  dbc.Nav([dbc.NavItem(dbc.NavLink("About", href="https://github.com/mchiusi/bye_splits")),
                           dbc.DropdownMenu([
                               dbc.DropdownMenuItem('3D view', id='3D view'),
                               dbc.DropdownMenuItem('Layer view', id='Layer view')],
                           label="Pages", nav=True)
                          ])
              ],brand="3D trigger cells visualization",color="primary",dark=True,)]),
    html.Br(),
    html.Div(id='page-content')
    ])

@app.callback(Output('page-content', 'children'),
              [Input('3D view','n_clicks'), Input('Layer view','n_clicks')])
def render_content(*args):
    button_id = ctx.triggered_id
    if button_id == '3D view' or not ctx.triggered:
        return processing.layout(checkbox=['Cluster trigger cells', 'ROI', 'Layer selection'], page='3D')
    elif button_id == 'Layer view':
        return processing.layout(checkbox=['Cluster trigger cells','Layer selection'], page='2D')

@app.callback([Output('event-display','children'),Output('out_slider','children'), 
              Output('dataframe','data'), Output('event','value')],
             [Input('particle','value'),Input('event-val','n_clicks'),
              Input('submit-val','n_clicks')],
             [State('event','value'), State('page', 'key')])
def update_event(particle, n_click, submit_event, event, page):
    button_clicked = ctx.triggered_id

    if button_clicked != 'submit-val':
        df, event, gen_info = processing.get_data(event=None, particles=particle)
    else:
        assert event != None, '''Please select manually an event or click on 'Random event'.'''
        df, event, gen_info = processing.get_data(event, particle)

    eta = gen_info['exeta'].values[0]
    phi = gen_info['exphi'].values[0]
    if particle == 'photons 200PU':
        x_gen, y_gen = processing.sph2cart(eta, phi)
        df = df[np.sqrt((x_gen-df.tc_x)**2+(y_gen-df.tc_y)**2)<100]
    
    if page == '3D':
        slider = dcc.RangeSlider(df['layer'].min(),df['layer'].max(), 
                             value=[df['layer'].min(), df['layer'].max()], step=None,
                             marks={int(layer) : {"label": str(layer)} for each, 
                                    layer in enumerate(sorted(df['layer'].unique()))}, 
                             id = 'slider')
    else:
        slider = dcc.Slider(df['layer'].min(),df['layer'].max(), 
                             value=11, step=None,
                             marks={int(layer) : {"label": str(layer)} for each, 
                                    layer in enumerate(sorted(df['layer'].unique()))}, 
                             id = 'slider')


    return u'Event {} selected. Gen Particle (\u03B7={:.2f}, \u03C6={:.2f}), {} GeV.'.format(int(event),gen_info['exeta'].values[0],gen_info['exphi'].values[0],int(gen_info['gen_energy'].values[0])), slider, df.reset_index().to_json(date_format='iso'), ''


@app.callback(Output('plot', 'figure'),  Output('slider-container', 'style'),
              [Input('dataframe', 'data'), Input('slider', 'value'), 
               Input('mip', 'value'), Input('checkbox', 'value')],
              [State('page', 'key')])
def make_graph(data, slider_value, mip, checkbox, page):
    assert float(mip) >= 0.5, 'mip\u209C value out of range. Minimum value 0.5 !'
    df = pd.read_json(data, orient='records')
    df_sel = df[df.mipPt >= mip]
    
    if page == '3D': 
        if 'Layer selection' in checkbox:
            df_sel = df_sel[(df_sel.layer >= slider_value[0]) & (df_sel.layer <= slider_value[1])]
    
        if 'Cluster trigger cells' in checkbox:
            df_no_cluster = df_sel[df_sel.tc_cluster_id == 0]
            df_cluster    = df_sel[df_sel.tc_cluster_id != 0]
            fig = processing.set_3dfigure(df_cluster)
            fig = processing.update_3dfigure(fig, df_no_cluster)
        else:
            fig = processing.set_3dfigure(df_sel)

        if 'ROI' in checkbox:
            fig = processing.add_ROI(fig, df_sel) 
    else:
        df_sel = df_sel[df_sel.layer == slider_value]

        if 'Cluster trigger cells' in checkbox: 
            df_no_cluster = df_sel[df_sel.tc_cluster_id == 0]
            df_cluster    = df_sel[df_sel.tc_cluster_id != 0]
            fig = processing.set_2dfigure(df_cluster)
            fig = processing.update_2dfigure(fig, df_no_cluster)
        else:
            fig = processing.set_2dfigure(df_sel)
   
    if 'Layer selection' not in checkbox and page != '2D':
        status_slider = {'display': 'none', 'width':'1'}
    else: 
        status_slider = {'display': 'block', 'width':'1'}
    return fig, status_slider


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',type=str,default='llruicms01.in2p3.fr', help='choice of host machine to use to run the application')
    parser.add_argument('--port',type=int,default=8004, help='choice of port to use for application')
    args = parser.parse_args()
    
    app.run_server(debug=True,
                   host=args.host,
                   port=args.port)
