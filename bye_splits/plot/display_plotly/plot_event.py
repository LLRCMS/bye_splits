# coding: utf-8

_all_ = []

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

def produce_3dplot(df, opacity=1, surfaceaxis=0):
    array_data = df[['diamond_x','diamond_y','z','waferu', 'waferv','colors','mipPt','energy']].to_numpy()
    listdata = []
    for j,i in enumerate(array_data):
        x1 = np.append(i[0],i[0][0])
        y1 = np.append(i[1],i[1][0])
        z1 = np.array(int(len(x1)) * [i[2]])
        datum = go.Scatter3d(x=z1, y=y1, z=x1, opacity=opacity,mode="lines", 
                            customdata=np.stack((5*[i[7]],5*[i[6]],5*[i[3]],5*[i[4]]), axis=-1),
                            hovertemplate='<b>Enegy[GeV]</b>: %{customdata[0]:,.2f}<br>' +
                                          '<b>mip\u209c</b>: %{customdata[1]:,.2f}<br>'+
                                          '<b>Wafer u</b>: %{customdata[2]}<br>'+
                                          '<b>Wafer v</b>: %{customdata[3]}<br>'+
                                          '<extra></extra>',
                            surfaceaxis=surfaceaxis,surfacecolor=i[5],marker=dict(color="black", showscale=True),
                            )
        listdata.append(datum)
    if opacity == 1:
        datum = go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(
                             colorscale='viridis',
                             cmin = df.mipPt.min(),
                             cmax = df.mipPt.max(),
                             showscale=True,
                             colorbar=dict(title=dict(text="[mip\u209c]", side="right"), ticks="outside", x=1)
                         ))
        listdata.append(datum)
    return listdata

def produce_2dplot(df, opacity=1):
    array_data = df[['diamond_x', 'diamond_y', 'tc_x', 'tc_y', 'colors','mipPt']].to_numpy()
    listdata = []
    for j,i in enumerate(array_data):
        x1 = np.append(i[0],i[0][0]) 
        y1 = np.append(i[1],i[1][0])
        datum = go.Scatter(x=x1, y=y1, opacity=opacity,mode="lines",fill='toself', fillcolor=i[4],
                          line_color='black',marker_line_color="black",  text=('Energy: '+str(round(i[5],2))))
        listdata.append(datum)
    return listdata


def plot_modules(df):
    array_data = df[['hex_x','hex_y','z']].to_numpy()
    listdata = []
    for j,i in enumerate(array_data):
        x1 = np.append(i[0],i[0][0])
        y1 = np.append(i[1],i[1][0])
        z1 = np.array(int(len(x1)) * [i[2]])
        datum = go.Scatter3d(x=z1, y=y1, z=x1, mode="lines", 
                            marker=dict(color="black"),
                            )
        listdata.append(datum)
    return listdata
