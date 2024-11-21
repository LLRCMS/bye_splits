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
import dash.dcc as dcc

from bye_splits.tasks.coarse_seeding import cs_dummy_calculator

def geom_selection(df_dict):
    gen_info = next(reversed(df_dict.values()))

    radius_gen = 40  # radius to select a specific region around the gen
    for chain in df_dict.keys():
        for coef in list(df_dict[chain].keys()):
            eta = gen_info['gen_eta'].values[0]
            phi = gen_info['gen_phi'].values[0]
            x_gen, y_gen = sph2cart(eta, phi)

            mask = np.sqrt((x_gen - df_dict[chain][coef]['tc_x'])**2 + \
                           (y_gen - df_dict[chain][coef]['tc_y'])**2) < radius_gen
            df_dict[chain][coef] = df_dict[chain][coef][mask].reset_index(drop=True)
    
    return df_dict

def prepare_slider(df, page='3D'):
    slider_options = {
        'min': df['tc_layer'].min(),
        'max': df['tc_layer'].max(),
        'step': None,
        'marks': {int(layer): {"label": str(int(layer))} 
                  for each, layer in enumerate(sorted(df['tc_layer'].unique()))},
        'id': 'slider'
    }
    
    if page == '3D':
        return dcc.RangeSlider(
            value=[df['tc_layer'].min(), df['tc_layer'].max()],
            **slider_options
        )
    else:
        return dcc.Slider(
            value=11,
            **slider_options
        )

def add_CS(fig, df):
    ''' Choose k-layers window based on energy deposited in each layer '''
    cs_df, _ = cs_dummy_calculator(df, threshold=[170, 180, 200])

    list_scatter = plot_modules(cs_df)
    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])

def sph2cart(eta, phi, z=322.):
    ''' Useful conversion '''
    theta = 2*np.arctan(np.exp(-eta))
    r = z / np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    return x, y

def _set_figure(fig, is_3d=False):
    axis = dict(
        backgroundcolor="rgba(0,0,0,0)",
        gridcolor="white",
        showbackground=True,
        zerolinecolor="white",
    )
    
    scene_params = dict(
        autosize=False,
        width=1300,
        height=700,
        scene_aspectmode='manual',
        scene_aspectratio=dict(x=1.3 if is_3d else 1, y=1, z=1),
        scene=dict(xaxis=axis, yaxis=axis, zaxis=axis if is_3d else None,
            xaxis_title="z [cm]" if is_3d else "y [cm]",
            yaxis_title="x [cm]" if is_3d else "x [cm]",
            zaxis_title="y [cm]" if is_3d else None,
            xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False,
        ),
        showlegend=False,
        uirevision=1,
        margin=dict(l=0, r=0, t=10, b=10),
    )

    fig.update_layout(**scene_params)

def set_figure(df, plot_type='2D', discrete=False):
    """ Create a Plotly figure with data from a DataFrame """
    fig = go.Figure(produce_plot(df, 1, plot_type, discrete))
    
    _set_figure(fig, is_3d=(plot_type=='3D'))
    return fig

def update_figure(fig, df, plot_type='2D', discrete=False):
    """ Update a Plotly figure with data from a DataFrame """
    list_scatter = produce_plot(df, 0.2, plot_type, discrete)

    for index in range(len(list_scatter)):
        fig.add_trace(list_scatter[index])

def produce_plot(df, opacity, plot_type, discrete):
    ''' Plot TCs from dataframe '''
    if plot_type == '2D':
        array_data = df[['diamond_x', 'diamond_y', 'tc_x', 'tc_y', 'colors', 'tc_mipPt']].to_numpy()
        x_column, y_column = 'diamond_x', 'diamond_y'
    elif plot_type == '3D':
        if discrete:
            array_data = df[['diamond_x', 'diamond_y', 'z', 'tc_wu', 'tc_wv', 'color_clusters', 'tc_mipPt']].to_numpy()
        else:
            array_data = df[['diamond_x', 'diamond_y', 'z', 'tc_wu', 'tc_wv', 'color_energy', 'tc_mipPt']].to_numpy()
        x_column, y_column = 'z', 'diamond_x'
    else:
        raise ValueError("Invalid plot_type. Choose '2D' or '3D'.")

    listdata = []
    for j, i in enumerate(array_data):
        x1 = np.append(i[0], i[0][0])
        y1 = np.append(i[1], i[1][0])
        if plot_type == '2D':
            datum = go.Scatter(x=x1, y=y1, opacity=opacity, mode="lines", fill='toself', fillcolor=i[4],
                               line_color='black', marker_line_color="black", text=('Energy: ' + str(round(i[5], 2))))
        else:
            z1 = np.full_like(x1, i[2])
            datum = go.Scatter3d(x=z1, y=x1, z=y1, opacity=opacity, mode="lines",
                                  customdata=np.stack((5 * [i[6]], 5 * [i[3]], 5 * [i[4]]), axis=-1),
                                  hovertemplate='<b>mip\u209c</b>: %{customdata[0]:,.2f}<br>' +
                                                '<b>Wafer u</b>: %{customdata[1]}<br>' +
                                                '<b>Wafer v</b>: %{customdata[2]}<br>' +
                                                '<extra></extra>',
                                  surfaceaxis=0, surfacecolor=i[5], marker=dict(color="black", showscale=True),
                                  )
        listdata.append(datum)

    if opacity == 1 and not discrete and plot_type == '3D':
        datum = go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", marker=dict(
            colorscale='viridis',
            cmin=df.tc_mipPt.min(),
            cmax=df.tc_mipPt.max(),
            showscale=True,
            colorbar=dict(title=dict(text='Transverse mip', font=dict(size=16), side="right"), ticks="outside", x=1)
        ))
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
