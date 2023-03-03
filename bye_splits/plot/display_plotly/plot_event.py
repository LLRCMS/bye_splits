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

def add_3dscintillators(fig, df):
    cart_coord = []
    x0, y0 = cil2cart(df['rmin'], df['phimin'])
    x1, y1 = cil2cart(df['rmax'], df['phimin'])
    x2, y2 = cil2cart(df['rmax'], df['phimax'])
    x3, y3 = cil2cart(df['rmin'], df['phimax'])

    d = np.array([x0.values, x1.values, x2.values, x3.values,
                  y0.values, y1.values, y2.values, y3.values]).T
    df_sci = pd.DataFrame(data=d, columns=['x1','x2','x3','x4','y1','y2','y3','y4'])
    df['diamond_y']= df_sci[['x1','x2','x3','x4']].values.tolist()
    df['diamond_x']= df_sci[['y1','y2','y3','y4']].values.tolist()

    list_scatter = produce_3dplot(df)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])

def add_ROI(fig, df, k=4):
    ''' Choose k-layers window based on energy deposited in each layer '''
    initial_layer = 7
    availLayers = [x-1 for x in cfg["selection"]["disconnectedTriggerLayers"]]
    mask = (df.layer>=initial_layer) & (df.layer<(availLayers[availLayers.index(initial_layer)+k]))
    input_df = df[mask]

    roi_df, module_ROI = roi_finder(input_df, threshold=30, nearby=False)

    list_scatter = plot_modules(roi_df)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])

################################################
############## Useful conversions ##############

# def sph2cart(eta, phi, z=322.):
#     theta = 2*np.arctan(np.exp(-eta))
#     r = z / np.cos(theta)
#     x = r * np.sin(theta) * np.cos(phi)
#     y = r * np.sin(theta) * np.sin(phi)
#     return x,y
# 
# def cil2cart(r, phi):
#     x = r * np.cos(phi)
#     y = r * np.sin(phi)
#     return x,y

def get_pt(energy, eta):
    return energy/np.cosh(eta)

###############################################################
############## Set scene for display and update  ##############

def set_3dfigure(df, discrete=False):
    fig = go.Figure(produce_3dplot(df,discrete=discrete))
     
    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white", showbackground=True, zerolinecolor="white")
    fig.update_layout(autosize=False, width=1300, height=700,
                      scene_aspectmode='manual',
                      scene_aspectratio=dict(x=1.3, y=1, z=1),
                      scene=dict(xaxis=axis, yaxis=axis, zaxis=axis,
                                 xaxis_title="z [cm]",yaxis_title="y [cm]",zaxis_title="x [cm]",
                                 xaxis_showspikes=False,yaxis_showspikes=False,zaxis_showspikes=False),
                      showlegend=False,
                      margin=dict(l=0, r=0, t=10, b=10),
                      ) 

    return fig

def update_3dfigure(fig, df, discrete=False):
    list_scatter = produce_3dplot(df, opacity=.2, discrete=discrete)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])

def set_2dfigure(df):
    fig = go.Figure(produce_2dplot(df))

    axis = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="white", showbackground=True, zerolinecolor="white",)
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


##########################################
############## Plot Objects ##############

def produce_3dplot(df, opacity=1, surfaceaxis=0, discrete=False):
    if discrete:
        array_data = df[['diamond_x','diamond_y','z','waferu','waferv','color_clusters','mipPt']].to_numpy()
    else:
        array_data = df[['diamond_x','diamond_y','z','waferu','waferv','color_energy','mipPt']].to_numpy()
    
    listdata = []
    for j,i in enumerate(array_data):
        x1 = np.append(i[0],i[0][0])
        y1 = np.append(i[1],i[1][0])
        z1 = np.full_like(x1,i[2])
        datum = go.Scatter3d(x=z1, y=y1, z=x1, opacity=opacity,mode="lines", 
                            customdata=np.stack((5*[i[6]],5*[i[3]],5*[i[4]]), axis=-1),
                            hovertemplate='<b>mip\u209c</b>: %{customdata[0]:,.2f}<br>'+
                                          '<b>Wafer u</b>: %{customdata[1]}<br>'+
                                          '<b>Wafer v</b>: %{customdata[2]}<br>'+
                                          '<extra></extra>',
                            surfaceaxis=surfaceaxis,surfacecolor=i[5],marker=dict(color="black", showscale=True),
                            )
        listdata.append(datum)

    if opacity == 1 and not discrete:
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
