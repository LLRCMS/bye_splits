# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import argparse
import numpy as np
import uproot as up

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
    Button,
    Slider,
    CustomJS,
    CustomJSFilter,
    CDSView,
    )
from bokeh.layouts import layout

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
        
def get_data():
    return handle('geom').provide(True)

# def get_data():
#     return handle('event').provide(True)

def display():
    doc = curdoc()
    source = ColumnDataSource(data=get_data())
    def update():
        source.data = get_data()

    width = int(1600/3)
    height = 400

    tc_vars = handle('geom').variables()
    
    slider  = Slider(start=source.data['layer'].min(), end=source.data['layer'].max(),
                     value=source.data['layer'].min(), step=2, title='Layer',
                     bar_color='red', default_size=800,
                     background='white')
    callback = CustomJS(args=dict(s=source), code="""s.change.emit();""")
    slider.js_on_change('value', callback) #value_throttled
        
    filt = CustomJSFilter(args=dict(slider=slider), code="""
           var indices = new Array(source.get_length());
           var sval = slider.value;
    
           const subset = source.data['layer'];
           for (var i=0; i < source.get_length(); i++) {
               indices[i] = subset[i] == sval;
           }
           return indices;
           """)
    view = CDSView(source=source, filters=[filt])
        
    p_uv = figure(width=width, height=height,
                  tools='save,reset', toolbar_location='right')
    p_uv.add_tools(BoxZoomTool(match_aspect=True))
    
    common_props(p_uv, xlim=(-20,20), ylim=(-20,20))
    p_uv.hex_tile(q=tc_vars['u'], r=tc_vars['vs'],
                  source=source, view=view,
                  size=1, fill_color='color',
                  line_color='black', line_width=1, alpha=1.)
            
    p_uv.add_tools(HoverTool(tooltips=[('u/v', '@'+tc_vars['u']+'/'+'@'+tc_vars['v']),]))
        
    # (x,y) plots
    p_xy = figure(width=width, height=height,
                  tools='save,reset', toolbar_location='right',
                  output_backend='webgl')
    p_xy.add_tools(BoxZoomTool(match_aspect=True))
    p_xy.add_tools(HoverTool(tooltips=[('u/v', '@'+tc_vars['u']+'/'+'@'+tc_vars['v']),],))
           
    common_props(p_xy, xlim=(-13,13), ylim=(-13,13))
    p_xy.rect(x=tc_vars['u'], y=tc_vars['v'],
              source=source, view=view,
              width=1., height=1.,
              width_units='data', height_units='data',
              fill_color='color',
              line_color='black',)
    
    # x VS z plots
    p_xVSz = figure(width=width, height=height,
                    tools='save,reset', toolbar_location='right')
    p_xVSz.add_tools(BoxZoomTool(match_aspect=True))
    #p_xy.add_tools(HoverTool(tooltips=[('u/v', '@u/@v'),],))
           
    #common_props(p_xy, xlim=(-13,13), ylim=(-13,13))
    p_xVSz.scatter(x=tc_vars['z'], y=tc_vars['x'], source=source)
        
    # y VS z plots
    p_yVSz = figure(width=width, height=height,
                    tools='save,reset', toolbar_location='right')
    p_yVSz.add_tools(BoxZoomTool(match_aspect=True))
    #p_xy.add_tools(HoverTool(tooltips=[('u/v', '@u/@v'),],))
           
    #common_props(p_xy, xlim=(-13,13), ylim=(-13,13))
    p_yVSz.scatter(x=tc_vars['z'], y=tc_vars['y'], source=source)
        
    # y VS x plots
    p_yVSx = figure(width=width, height=height,
                    tools='save,reset', toolbar_location='right')
    p_yVSx.add_tools(BoxZoomTool(match_aspect=True))
    #p_xy.add_tools(HoverTool(tooltips=[('u/v', '@u/@v'),],))
           
    #common_props(p_xy, xlim=(-13,13), ylim=(-13,13))
    p_yVSx.scatter(x=tc_vars['x'], y=tc_vars['y'], source=source)
        
    button = Button(label='Update', button_type='success', default_size=100)
    button.on_click(update)

    blank1 = Div(width=1000, height=100, text='')
    blank2 = Div(width=70, height=100, text='')

    lay = layout([[button, blank2, slider],
                  [p_uv,p_xy],
                  [blank1],
                  [p_xVSz,p_yVSz,p_yVSx]])
    doc.add_root(lay) # save(lay)
    doc.title = 'TC Visualization'

parser = argparse.ArgumentParser(description='')
FLAGS = parser.parse_args()
display()
