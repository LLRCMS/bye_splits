# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

from dash import Dash, dcc, html, Input, Output
import plotly.express as px

import numpy as np
import data_handle
from data_handle.data_handle import handle

app = Dash(__name__)
app.title = '3D Visualization' 

def get_data():
    return handle('geom').provide(True)

df = get_data()
mask = df.layer >= 0

fig = px.scatter_3d(df[mask], 
                    x='z', y='x', z='y',
                    #color='color',
                    color_discrete_sequence=['black'],
                    #symbol='*',
                    symbol_sequence=['circle'],
                    hover_data=['x', 'y', 'z'],
                    )
fig.update_traces(marker=dict(size=1.,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
fig.update_layout(autosize=False, width=2100, height=1200,
                  scene_aspectmode='manual',
                  scene_aspectratio=dict(x=4, y=1, z=1),
                  scene=dict(xaxis=dict(backgroundcolor="rgba(0,0,0,0)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                             yaxis=dict(backgroundcolor="rgba(0,0,0,0)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white"),
                             zaxis=dict(backgroundcolor="rgba(0,0,0,0)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),),
                  )

app.layout = html.Div(children=[
    html.H4('3D TC distribution'),
    dcc.Graph(id='graph', figure=fig),
    #html.P('More info'),
    # dcc.RangeSlider(
    #     id='range-slider',
    #     min=0, max=2.5, step=0.1,
    #     marks={0: '0', 2.5: '2.5'},
    #     value=[0.5, 2]
    # ),
    ])

if __name__ == '__main__':
    app.run_server(debug=True,
                   host='llruicms01.in2p3.fr',
                   port=8004)

# @app.callback(
#     Output('graph', 'figure'), 
#     #Input('range-slider', 'value')
#     Input('range-slider', 'value')
#     )
# def update_bar():
#     df = get_data()
#     # low, high = slider_range
#     mask = np.ones_like(df.x)

#     fig = px.scatter_3d(df[mask], 
#         x='x', y='y', z='z', color='c', hover_data=['x', 'y', 'z'])
#     return fig
