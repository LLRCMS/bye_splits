# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

from functools import partial
import argparse
import numpy as np
import pandas as pd
import uproot as up
import awkward as ak

#from bokeh.io import output_file, save
#output_file('tmp.html')
from bokeh.plotting import figure, curdoc
from bokeh.util.hex import axial_to_cartesian
from bokeh.models import (
    Div,
    Tabs,
    BoxZoomTool,
    Range1d,
    ColumnDataSource,
    HoverTool,
    TextInput,
    TabPanel,
    Tabs,
    Slider,
    CustomJS,
    CustomJSFilter,
    CDSView,
    WheelZoomTool,
    )
from bokeh.layouts import layout
from bokeh.settings import settings
settings.ico_path = 'none'

import utils
from utils import params, common, parsing
import data_handle
from data_handle.data_handle import handle

def common_props(p, xlim=None, ylim=None):
    p.output_backend = 'svg'
    p.toolbar.logo = None
    p.grid.visible = False
    p.outline_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False
    if xlim is not None:
        p.x_range = Range1d(xlim[0], xlim[1])
    if ylim is not None:
        p.y_range = Range1d(ylim[0], ylim[1])
        
def convert_cells_to_xy(df, avars):
    d, d4 = 1, 1
    cos30, sin30 = np.sqrt(3)/2, 1/2
    conversion = {'UL': (lambda v: cos30 * d4 * v,
                        lambda u,v: sin30 * d4 * (2*(u-1)-v)),
                  'UR': (lambda v: 4*cos30 + cos30 * d4 * (v-4),
                         lambda u,v: 2 + sin30 * d4 * (2*(u-4)-(v-4))),
                  'B':  (lambda v: cos30 * d4 * v,
                         lambda u,v: sin30 * d4 * (2*u-v))
                         } #up-right, up-left and bottom

    masks = {'UL': (((df[avars['cu']]>=1) & (df[avars['cu']]<=4) & (df[avars['cv']]==0)) |
                    ((df[avars['cu']]>=2) & (df[avars['cu']]<=5) & (df[avars['cv']]==1)) |
                    ((df[avars['cu']]>=3) & (df[avars['cu']]<=6) & (df[avars['cv']]==2)) |
                    ((df[avars['cu']]>=4) & (df[avars['cu']]<=7) & (df[avars['cv']]==3))),
             'UR': (df[avars['cu']]>=4) & (df[avars['cu']]<=7) & (df[avars['cv']]>=4) & (df[avars['cv']]<=7),
             'B':  (df[avars['cv']]>=df[avars['cu']]) & (df[avars['cu']]<=3),
             }
     
    x0, x1, x2, x3 = ({} for _ in range(4))
    y0, y1, y2, y3 = ({} for _ in range(4))
    xaxis, yaxis = ({} for _ in range(2))
    for key,val in masks.items():
        x0.update({key: conversion[key][0](df[avars['cv']][masks[key]])})
        x1.update({key: x0[key][:] + cos30})
        if key in ('UL', 'UR'):
            x2.update({key: x1[key][:]})
            x3.update({key: x0[key][:]})
        else:
            x2.update({key: x1[key][:] + cos30})
            x3.update({key: x1[key][:]})
     
        y0.update({key: conversion[key][1](df[avars['cu']][masks[key]],
                                           df[avars['cv']][masks[key]])})
        if key in ('UR', 'B'):
            y1.update({key: y0[key][:] - sin30})
        else:
            y1.update({key: y0[key][:] + sin30})
        if key in ('B'):
            y2.update({key: y0[key][:]})
        else:
            y2.update({key: y1[key][:] + d})
        if key in ('UL', 'UR'):
            y3.update({key: y0[key][:] + d})
        else:
            y3.update({key: y0[key][:] + sin30})

        xaxis.update({key: np.stack([x0[key],x1[key],x2[key],x3[key]],
                                    axis=1).reshape((-1,4))})
        yaxis.update({key: np.stack([y0[key],y1[key],y2[key],y3[key]],
                                    axis=1).reshape((-1,4))})

    tc_polyg_x = np.concatenate([v for v in xaxis.values()], axis=0)
    tc_polyg_x = pd.DataFrame(tc_polyg_x).groupby(df.index).agg(list).sum(axis=1).agg(lambda x: [[x]])
    
    tc_polyg_y = np.concatenate([v for v in yaxis.values()], axis=0)
    tc_polyg_y = pd.DataFrame(tc_polyg_y).groupby(df.index).agg(list).sum(axis=1).agg(lambda x: [[x]])
    res = pd.concat([tc_polyg_x, tc_polyg_y], axis=1)
    res.columns = ['tc_polyg_x', 'tc_polyg_y']
    df.rename(columns = {avars['l']: 'layer'}, inplace=True)
    return df.join(res)

def get_data(particle):
    event = 179855 if particle == 'photons' else 92004
    ev_vars = handle('event', particle).variables()
    geom_vars = handle('geom').variables()

    ds_ev = handle('event', particle).provide_event(event, True)
    ds_ev = convert_cells_to_xy(ds_ev, ev_vars)
    ds_geom = handle('geom').provide(True)
    ds_geom = convert_cells_to_xy(ds_geom, geom_vars)
    breakpoint()
    return ds_ev, ds_geom 

sources = {'photons'   : ColumnDataSource(data=get_data('photons')[1]),
           'electrons' : ColumnDataSource(data=get_data('electrons')[1]),
           }
elements = {'photons'   : {'textinput': TextInput(title='Event', value='', sizing_mode='stretch_width')},
            'electrons' : {'textinput': TextInput(title='Event', value='', sizing_mode='stretch_width')},
            }
assert sources.keys() == elements.keys()

def display():
    def text_callback(attr, old, new, source, particle):
        print('running ', particle)
        source.data = get_data(particle)[1]

    doc = curdoc()
    doc.title = 'TC Visualization'
    
    width, height = int(1600/3), 400
    tabs = []
    
    for ksrc,vsrc in sources.items():
        ev_vars = handle('event', ksrc).variables()
        geom_vars = handle('geom').variables()

        slider = Slider(start=vsrc.data['layer'].min(), end=vsrc.data['layer'].max(),
                        value=vsrc.data['layer'].min(), step=2, title='Layer',
                        bar_color='red', width=800, background='white')
        slider_callback = CustomJS(args=dict(s=vsrc), code="""s.change.emit();""")
        slider.js_on_change('value', slider_callback) #value_throttled

        view = CDSView(filter=CustomJSFilter(args=dict(slider=slider), code="""
           var indices = new Array(source.get_length());
           var sval = slider.value;
    
           const subset = source.data['layer'];
           for (var i=0; i < source.get_length(); i++) {
               indices[i] = subset[i] == sval;
           }
           return indices;
           """))

        ####### (u,v) plots ################################################################
        # p_uv = figure(width=width, height=height,
        #               tools='save,reset', toolbar_location='right')
        # p_uv.add_tools(WheelZoomTool(),
        #                BoxZoomTool(match_aspect=True))
        # common_props(p_uv, xlim=(-20,20), ylim=(-20,20))
        # p_uv.hex_tile(q=ev_vars['tcwu'], r=ev_vars['tcwv'], source=vsrc, view=view,
        #               size=1, fill_color='color', line_color='black', line_width=1, alpha=1.)    
        # p_uv.add_tools(HoverTool(tooltips=[('u/v', '@'+ev_vars['tcwu']+'/'+'@'+ev_vars['tcwv']),]))

        ####### cell plots ################################################################
        polyg_opt = dict(line_color='black', line_width=3)
        p_cells = figure(width=width, height=height,
                         tools='save,reset', toolbar_location='right',
                         output_backend='webgl')
        p_cells.add_tools(BoxZoomTool(match_aspect=True),
                          HoverTool(tooltips=[('u/v', '@'+geom_vars['cu']+'/'+'@'+geom_vars['cv']),]))
        p_cells.multi_polygons(xs='tc_polyg_x', ys='tc_polyg_y',
                               source=vsrc, view=view, color='red',
                               **polyg_opt)
        
        ####### (x,y) plots ################################################################
        # p_xy = figure(width=width, height=height,
        #             tools='save,reset', toolbar_location='right',
        #             output_backend='webgl')
        # p_xy.add_tools(WheelZoomTool(), BoxZoomTool(match_aspect=True))
        # p_xy.add_tools(HoverTool(tooltips=[('u/v', '@'+ev_vars['tcwu']+'/'+'@'+ev_vars['tcwv']),],))       
        # common_props(p_xy, xlim=(-13,13), ylim=(-13,13))
        # p_xy.rect(x=ev_vars['tcwu'], y=ev_vars['tcwv'], source=vsrc, view=view,
        #           width=1., height=1., width_units='data', height_units='data',
        #           fill_color='color', line_color='black',)

        ####### x vs. z plots ################################################################
        # p_xVSz = figure(width=width, height=height, tools='save,reset', toolbar_location='right')
        # p_xVSz.add_tools(BoxZoomTool(match_aspect=True))
        # p_xVSz.scatter(x=ev_vars['z'], y=ev_vars['x'], source=vsrc)
        # common_props(p_xVSz)
        
        ####### y vs. z plots ################################################################
        # p_yVSz = figure(width=width, height=height, tools='save,reset', toolbar_location='right')
        # p_yVSz.add_tools(BoxZoomTool(match_aspect=True))
        # p_yVSz.scatter(x=ev_vars['z'], y=ev_vars['y'], source=vsrc)
        # common_props(p_yVSz)
        
        ####### y vs. x plots ################################################################
        # p_yVSx = figure(width=width, height=height, tools='save,reset', toolbar_location='right')
        # p_yVSx.add_tools(BoxZoomTool(match_aspect=True))
        # p_yVSx.scatter(x=ev_vars['x'], y=ev_vars['y'], source=vsrc)
        # common_props(p_yVSx)
        
        ####### text input ###################################################################
        elements[ksrc]['textinput'].on_change('value', partial(text_callback, particle=ksrc, source=vsrc))

        ####### define layout ################################################################
        blank1 = Div(width=1000, height=100, text='')
        blank2 = Div(width=70, height=100, text='')

        lay = layout([[elements[ksrc]['textinput'], blank2, slider],
                      #[p_cells, p_uv, p_xy],
                      [p_cells],
                      [blank1],
                      # [p_xVSz,p_yVSz,p_yVSx]
                      ])
        tab = TabPanel(child=lay, title=ksrc)
        tabs.append(tab)
        # end for loop

    doc.add_root(Tabs(tabs=tabs))
    
parser = argparse.ArgumentParser(description='')
FLAGS = parser.parse_args()
display()

# (Pdb) p ds_geom.loc[240635]              
# waferu                                                         -2
# waferv                                                         -5
# layer                                                           5
# triggercellu                                                    6
# triggercellv                                                    4
# x                                                       -7.216201
# y                                                      -67.298866
# z                                                      328.042755
# waferv_shift                                                    5
# color                                                     #8a2be2
# tc_polyg_x      [[[2.598076211353316, 3.4641016151377544, 3.46...
# tc_polyg_y                               [[[4.5, 5.0, 6.0, 5.5]]]
# Name: 240635, dtype: object
# (Pdb) p ds_geom.loc[240730]
# waferu                                                         -4
# waferv                                                        -11
# layer                                                           5
# triggercellu                                                    7
# triggercellv                                                    5
# x                                                       -21.82198
# y                                                     -153.202576
# z                                                      328.042755
# waferv_shift                                                   11
# color                                                     #8a2be2
# tc_polyg_x      [[[2.598076211353316, 3.4641016151377544, 3.46...
# tc_polyg_y                               [[[4.5, 5.0, 6.0, 5.5]]]
# Name: 240730, dtype: object
