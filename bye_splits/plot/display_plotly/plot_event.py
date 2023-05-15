# coding: utf-8

_all_ = []

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
from dash import dcc
import plotly.graph_objects as go

def geom_selection(df_dict):
    gen_info = next(reversed(df_dict.values()))

    for coef in list(df_dict.keys())[:-1]:
        eta = gen_info['gen_eta'].values[0]
        phi = gen_info['gen_phi'].values[0]
        x_gen, y_gen = sph2cart(eta, phi)
        df_dict[coef] = df_dict[coef][np.sqrt((x_gen-df_dict[coef].tc_x)**2+(y_gen-df_dict[coef].tc_y)**2)<50]

    return df_dict

def prepare_slider(df, page='3D'):
    if page == '3D':
        return dcc.RangeSlider(df['layer'].min(),df['layer'].max(),value=[df['layer'].min(), 
                               df['layer'].max()], step=None, marks={int(layer) : {"label": str(int(layer))} for each,
                               layer in enumerate(sorted(df['layer'].unique()))}, id = 'slider')
    else:
        return dcc.Slider(df['layer'].min(),df['layer'].max(),value=11, step=None,
                          marks={int(layer) : {"label": str(layer)} for each,
                          layer in enumerate(sorted(df['layer'].unique()))}, id = 'slider')

def add_ROI(fig, df, k=4):
    ''' Choose k-layers window based on energy deposited in each layer '''
    initial_layer = 7
    availLayers = [x for x in range(50) if x % 2 != 0]
    mask = (df.layer>=initial_layer) & (df.layer<(availLayers[availLayers.index(initial_layer)+k]))
    input_df = df[mask]

    roi_df, module_ROI = roi_finder_thr(input_df, threshold=[170, 180, 200])

    list_scatter = plot_modules(roi_df)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])

def roi_finder_thr(input_df, threshold=[170, 180, 200], eta_list=[1.5, 1.9, 2.3, 3.2], nearby=False):
   ''' Choose the (u,v) coordinates of the module corresponding to max dep-energy.
   This choice is performed by grouping modules beloging to different layers, having the same coordinates.
   Extend the selection to the nearby modules with at least 30% max energy and 10 mipT. '''
   module_sums = input_df.groupby(['waferu','waferv']).mipPt.sum()
   eta_coord = input_df.groupby(['waferu','waferv']).tc_eta.mean()

   module_ROI = []
   for index in range(len(eta_list)-1):
       modules = list(module_sums[(module_sums.values >= threshold[index]) &
                                  (eta_coord.values <= eta_list[index+1])  &
                                  (eta_coord.values > eta_list[index])].index)
       module_ROI.extend(modules)
   
   if nearby:
       selected_modules = []
       for module in module_ROI:
           nearby_modules = [(module[0]+i, module[1]+j) for i in [-1, 0, 1] for j in [-1, 0, 1] if i*j >= 0]
           
           skim = (module_sums.index.isin(nearby_modules)) & (module_sums.values > 0.6*module_sums[module])
           skim_nearby_module = list(module_sums[skim].index)
           selected_modules.extend(skim_nearby_module)
       module_ROI = selected_modules
   
   roi_df = input_df[input_df.set_index(['waferu','waferv']).index.isin(module_ROI)]
   return roi_df, module_ROI

################################################
############## Useful conversions ##############

def sph2cart(eta, phi, z=322.):
    theta = 2*np.arctan(np.exp(-eta))
    r = z / np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    return x,y

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
                                 xaxis_title="z [cm]",yaxis_title="x [cm]",zaxis_title="y [cm]",
                                 xaxis_showspikes=False,yaxis_showspikes=False,zaxis_showspikes=False),
                      showlegend=False,
                      uirevision=1,
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
                                 xaxis_title="y [cm]",yaxis_title="x [cm]",
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
        datum = go.Scatter3d(x=z1, y=x1, z=y1, opacity=opacity,mode="lines", 
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
        datum = go.Scatter3d(x=z1, y=x1, z=y1, mode="lines", 
                            marker=dict(color="black"),
                            )
        listdata.append(datum)
    return listdata
