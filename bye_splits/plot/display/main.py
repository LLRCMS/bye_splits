
# coding: utf-8

_all_ = [ ]

import os
import pathlib
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

from functools import partial
import argparse
import numpy as np
import pandas as pd

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
    BasicTicker,
    ColorBar, 
    LogColorMapper,
    LogTicker,
    LinearColorMapper,
    BasicTicker,
    PrintfTickFormatter,
    )
from bokeh.palettes import viridis as _palette
mypalette = _palette(50)
from bokeh.layouts import layout
from bokeh.settings import settings
settings.ico_path = 'none'

import utils
from utils import params, common, parsing
import data_handle
from data_handle.data_handle import handle
data_vars = {'ev': handle('event').variables,
             'geom': handle('geom').variables,
             }
mode = 'geom'

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
    c30, s30, t30 = np.sqrt(3)/2, 1/2, 1/np.sqrt(3)
    d = 1.#1./c30
    d4, d8 = d/4., d/8.
    conversion = {
        'UL': (lambda wu,wv,cv: 2*wu - wv + d4 * cv,
               lambda wv,cu,cv: ((d/c30)+d*t30)*wv + d8/c30 * (2*(cu-1)-cv)),
        'UR': (lambda wu,wv,cv: 2*wu - wv + d + d4 * (cv-4),
               lambda wv,cu,cv: ((d/c30)+d*t30)*wv + t30*d + d8/c30 * (2*(cu-4)-(cv-4))),
        'B':  (lambda wu,wv,cv: 2*wu - wv + d4 * cv,
               lambda wv,cu,cv: ((d/c30)+d*t30)*wv + t30 * d4 * (2*cu-cv))
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
        x0.update({key: conversion[key][0](df[avars['wu']][masks[key]],
                                           df[avars['wv']][masks[key]],
                                           df[avars['cv']][masks[key]])})
        x1.update({key: x0[key][:] + d4})
        if key in ('UL', 'UR'):
            x2.update({key: x1[key][:]})
            x3.update({key: x0[key][:]})
        else:
            x2.update({key: x1[key][:] + d4})
            x3.update({key: x1[key][:]})
     
        y0.update({key: conversion[key][1](df[avars['wv']][masks[key]],
                                           df[avars['cu']][masks[key]],
                                           df[avars['cv']][masks[key]])})
        if key in ('UR', 'B'):
            y1.update({key: y0[key][:] - t30*d4})
        else:
            y1.update({key: y0[key][:] + t30*d4})
        if key in ('B'):
            y2.update({key: y0[key][:]})
        else:
            y2.update({key: y1[key][:] + d4/c30})
        if key in ('UL', 'UR'):
            y3.update({key: y0[key][:] + d4/c30})
        else:
            y3.update({key: y0[key][:] + t30*d4})

        xaxis.update({key: pd.concat([x0[key],x1[key],x2[key],x3[key]], axis=1)})
        yaxis.update({key: pd.concat([y0[key],y1[key],y2[key],y3[key]], axis=1)})
        xaxis[key]['new'] = xaxis[key].values.tolist()
        yaxis[key]['new'] = yaxis[key].values.tolist()
        xaxis[key] = xaxis[key]['new']
        yaxis[key] = yaxis[key]['new']

    tc_polyg_x = pd.concat(xaxis.values())
    tc_polyg_y = pd.concat(yaxis.values())
    tc_polyg_x = tc_polyg_x.groupby(tc_polyg_x.index).agg(lambda k: [k])
    tc_polyg_y = tc_polyg_y.groupby(tc_polyg_y.index).agg(lambda k: [k])

    res = pd.concat([tc_polyg_x, tc_polyg_y], axis=1)
    res.columns = ['tc_polyg_x', 'tc_polyg_y']
    df.rename(columns = {avars['l']: 'layer'}, inplace=True)
    return df.join(res)

def get_data(event, particle):
    #event = 179855 if particle == 'photons' else 92004
    ds_ev = handle('event', particle).provide_event(event, True)
    ds_ev = convert_cells_to_xy(ds_ev, data_vars['ev'])
    ds_geom = handle('geom').provide(True)
    ds_geom = convert_cells_to_xy(ds_geom, data_vars['geom'])
    ds_geom = ds_geom[((ds_geom[data_vars['geom']['wu']]==4) & (ds_geom[data_vars['geom']['wv']]==4) |
                       (ds_geom[data_vars['geom']['wu']]==5) & (ds_geom[data_vars['geom']['wv']]==4) |
                       (ds_geom[data_vars['geom']['wu']]==5) & (ds_geom[data_vars['geom']['wv']]==5))]
    return {'ev': ds_ev, 'geom': ds_geom}

sources = {'photons'   : ColumnDataSource(data=get_data(179855, 'photons')[mode]),
           'electrons' : ColumnDataSource(data=get_data(92004,  'electrons')[mode]),
           }
elements = {'photons'   : {'textinput': TextInput(title='Event', value='', sizing_mode='stretch_width')},
            'electrons' : {'textinput': TextInput(title='Event', value='', sizing_mode='stretch_width')},
            }

assert sources.keys() == elements.keys()

def display():
    variables = data_vars[mode]
    def text_callback(attr, old, new, source, particle):
        print('running ', particle, new)
        assert new.isdecimal()
        source.data = get_data(new, particle)[mode]

    doc = curdoc()
    doc.title = 'TC Visualization'
    
    width, height   = 1250, 1000
    width2, height2 = 400, 300
    tabs = []
    
    for ksrc,vsrc in sources.items():
        if mode == 'ev':
            mapper = LinearColorMapper(palette=mypalette,
                                       low=vsrc.data[variables['en']].min(), high=vsrc.data[variables['en']].min())

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
        # p_uv.hex_tile(q=variables['tcwu'], r=variables['tcwv'], source=vsrc, view=view,
        #               size=1, fill_color='color', line_color='black', line_width=1, alpha=1.)    
        # p_uv.add_tools(HoverTool(tooltips=[('u/v', '@'+variables['tcwu']+'/'+'@'+variables['tcwv']),]))

        ####### cell plots ################################################################
        lim = 22
        polyg_opt = dict(line_color='black', line_width=2)
        p_cells = figure(width=width, height=height,
                         x_range=Range1d(-lim, lim),
                         y_range=Range1d(-lim, lim),
                         tools='save,reset', toolbar_location='right',
                         output_backend='webgl')
        if mode == 'ev':
            hover_key = 'Energy (cu,cv / wu,wv)'
            hover_val = '@'+variables['en'] + ' (@'+variables['cu']+' / @'+variables['cv']+',@'+variables['wu']+',@'+variables['wv']+')'
        else:
            hover_key = 'cu,cv / wu,wv'
            hover_val = '@'+variables['cu']+',@'+variables['cv']+' / @'+variables['wu']+',@'+variables['wv']

        p_cells.add_tools(BoxZoomTool(match_aspect=True),
                          WheelZoomTool(),
                          HoverTool(tooltips=[(hover_key, hover_val),]))
        common_props(p_cells, xlim=(-lim, lim), ylim=(-lim, lim))
        p_cells_opt = dict(xs='tc_polyg_x', ys='tc_polyg_y',
                           source=vsrc, view=view, **polyg_opt)
        
        if mode == 'ev':
            p_cells.multi_polygons(fill_color={'field': variables['en'], 'transform': mapper},
                                   **p_cells_opt)
        else:
            p_cells.multi_polygons(color='green', **p_cells_opt)
                        
        if mode == 'ev':
            color_bar = ColorBar(color_mapper=mapper,
                                ticker= BasicTicker(desired_num_ticks=int(len(mypalette)/4)),
                                formatter=PrintfTickFormatter(format="%d"))
            p_cells.add_layout(color_bar, 'right')

        ####### (x,y) plots ################################################################
        # p_xy = figure(width=width, height=height,
        #             tools='save,reset', toolbar_location='right',
        #             output_backend='webgl')
        # p_xy.add_tools(WheelZoomTool(), BoxZoomTool(match_aspect=True))
        # p_xy.add_tools(HoverTool(tooltips=[('u/v', '@'+variables['tcwu']+'/'+'@'+variables['tcwv']),],))       
        # common_props(p_xy, xlim=(-13,13), ylim=(-13,13))
        # p_xy.rect(x=variables['tcwu'], y=variables['tcwv'], source=vsrc, view=view,
        #           width=1., height=1., width_units='data', height_units='data',
        #           fill_color='color', line_color='black',)

        # ####### x vs. z plots ################################################################
        # p_xVSz = figure(width=width2, height=height2, tools='save,reset', toolbar_location='right')
        # p_xVSz.add_tools(BoxZoomTool(match_aspect=True))
        # p_xVSz.scatter(x=variables['z'], y=variables['x'], source=vsrc)
        # common_props(p_xVSz)
        
        # ####### y vs. z plots ################################################################
        # p_yVSz = figure(width=width2, height=height2, tools='save,reset', toolbar_location='right')
        # p_yVSz.add_tools(BoxZoomTool(match_aspect=True))
        # p_yVSz.scatter(x=variables['z'], y=variables['y'], source=vsrc)
        # common_props(p_yVSz)
        
        # ####### y vs. x plots ################################################################
        # p_yVSx = figure(width=width2, height=height2, tools='save,reset', toolbar_location='right')
        # p_yVSx.add_tools(BoxZoomTool(match_aspect=True))
        # p_yVSx.scatter(x=variables['x'], y=variables['y'], source=vsrc)
        # common_props(p_yVSx)
        
        ####### text input ###################################################################
        elements[ksrc]['textinput'].on_change('value', partial(text_callback, particle=ksrc, source=vsrc))

        ####### define layout ################################################################
        blank1 = Div(width=1000, height=100, text='')
        blank2 = Div(width=70, height=100, text='')

        lay = layout([[elements[ksrc]['textinput'], blank2, slider],
                      #[p_cells, p_uv, p_xy],
                      #[p_xVSz, p_yVSz, p_yVSx],
                      [p_cells],
                      [blank1],
                      ])
        tab = TabPanel(child=lay, title=ksrc)
        tabs.append(tab)
        # end for loop

    doc.add_root(Tabs(tabs=tabs))
    
parser = argparse.ArgumentParser(description='')
FLAGS = parser.parse_args()
display()
