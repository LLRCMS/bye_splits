# coding: utf-8

_all_ = [ ]

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * '/..')
sys.path.insert(0, parent_dir)

from dash import Dash, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
from dash_bootstrap_templates import load_figure_template
import dash_bootstrap_components as dbc
import pandas as pd
import argparse

from bye_splits.utils import parsing, common
import plot_event as plt
import processing

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = '3D Visualization' 
app.config['suppress_callback_exceptions'] = True
load_figure_template('FLATLY')
        
app.layout = html.Div([
    html.Div([dbc.NavbarSimple([
                  dbc.Nav([dbc.DropdownMenu([
                               dbc.DropdownMenuItem('3D view', id='3D view'),
                               dbc.DropdownMenuItem('Layer view', id='Layer view')],
                           label="Pages", nav=True)
                          ])
              ],brand="3D trigger cells visualization",color="primary",dark=True,)]),
    html.Br(),
    html.Div(id='page-content')
    ])


@app.callback(
    Output('page-content', 'children'),
    [Input('3D view', 'n_clicks'), Input('Layer view', 'n_clicks')]
)
def render_content(*args):
    button_id = ctx.triggered_id
    if button_id == '3D view' or not ctx.triggered:
        return process.layout(page='3D')
    elif button_id == 'Layer view':
        return process.layout(page='2D')


@app.callback(
    [
        Output('event-display', 'children'),
        Output('checkbox', 'options'),
        Output('out_slider', 'children'),
        Output('dataframe', 'data'),
        Output('event', 'value'),
    ],
    [
        Input('particle', 'value'),
        Input('pu', 'value'),
        Input('event-val', 'n_clicks'),
        Input('submit-val', 'n_clicks'),
    ],
    [
        State('event', 'value'),
        State('page', 'key'),
    ]
)
def update_event(particle, pu, n_click, submit_event, event, page):
    button_clicked = ctx.triggered_id
    if button_clicked == 'submit-val':
        if event is None:  raise RuntimeError("Please select manually an event or click on 'Random event'.")

    df_dict, event = process.get_data(common.dot_dict(vars(args)), particle, pu, event)
    
    gen_info = next(reversed(df_dict.values()))
    slider = plt.prepare_slider(next(iter(df_dict.values())), page)

    if particle == 'photons' and pu == '200 PU':
        df_dict = plt.geom_selection(df_dict)
    if particle != 'pions': checkbox = ['Coarse seeding','Layer selection','Seed index']
    else: checkbox = ['Layer selection','Seed index']

    df_dict = {k: v.to_json() for k, v in df_dict.items()}
    event_selected_message = 'Event {} selected. Gen Particle (η={:.2f}, ϕ={:.2f}), pₜ={:.2f} GeV.'
    return (event_selected_message.format(int(event), gen_info['gen_eta'].values[0], 
            gen_info['gen_phi'].values[0], gen_info['gen_pt'].values[0]), 
            checkbox, slider, df_dict, None)

@app.callback(
    [
        Output('plot', 'figure'),
        Output('slider-container', 'style')
    ],
    [
        Input('dataframe', 'data'),
        Input('slider', 'value'),
        Input('slider_cluster', 'value'),
        Input('mip', 'value'),
        Input('checkbox', 'value'),
    ],
    [State('page', 'key')]
)
def make_graph(data, slider_value, coef, mip, checkbox, page):
    assert float(mip) >= 0.5, 'mip\u209C value out of range. Minimum value 0.5 !'
    df_dict = {k: pd.read_json(v) for k, v in data.items()}
    df = df_dict[str(coef)]
    df_sel = df[df.tc_mipPt >= mip]

    if page == '3D':
        if 'Layer selection' in checkbox:
            df_sel = df_sel[(df_sel.tc_layer >= slider_value[0]) & (df_sel.tc_layer <= slider_value[1])]
        
        if 'Seed index' in checkbox: discrete = True
        else: discrete = False
        df_no_cluster = df_sel[df_sel['seed_idx'] == df_sel['seed_idx'].max()]
        df_cluster    = df_sel[df_sel['seed_idx'] != df_sel['seed_idx'].max()]
        fig = plt.set_figure(df_cluster, '3D', discrete)
        plt.update_figure(fig, df_no_cluster, '3D', discrete)

        if 'Coarse seeding' in checkbox:
            plt.add_CS(fig, df_sel) 
    else:
        df_sel = df_sel[df_sel.tc_layer == slider_value]

        if 'Cluster trigger cells' in checkbox: 
            df_no_cluster = df_sel[df_sel.tc_cluster_id == 0]
            df_cluster    = df_sel[df_sel.tc_cluster_id != 0]
            fig = plt.set_figure(df_cluster, '2D')
            plt.update_figure(fig, df_no_cluster, '2D')
        else:
            fig = plt.set_figure(df_sel, '2D')
  
    if 'Layer selection' not in checkbox and page != '2D':
        status_slider = {'display': 'none', 'width':'1'}
    else: 
        status_slider = {'display': 'block', 'width':'1'}
    return fig, status_slider


if __name__ == '__main__':
    args = parsing.parser_display_plotly()
    process = processing.Processing() 
    
    app.run_server(host=args.host, port=args.port, debug=args.debug)
