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
    TextInput,
    TabPanel,
    Tabs,
    Slider,
    CustomJS,
    CustomJSFilter,
    CDSView,
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
        
def get_data(particle):
    ds = handle('event', particle).provide(True)
    return ds, handle('geom').provide(True) 

sources = {'photons'   : ColumnDataSource(data=get_data('photons')[0]),
           'electrons' : ColumnDataSource(data=get_data('electrons')[0]),
           }
elements = {'photons'   : {'textinput': TextInput(title='Event', value='', sizing_mode='stretch_width')},
            'electrons' : {'textinput': TextInput(title='Event', value='', sizing_mode='stretch_width')},
            }
assert sources.keys() == elements.keys()

def display():
    def text_callback(attr, old, new, source, particle):
        print('running ', particle)
        source.data = get_data(particle)[0]

    doc = curdoc()
    doc.title = 'TC Visualization'
    
    width, height = int(1600/3), 400
    tabs = []
    
    for ksrc,vsrc in sources.items():
        tc_vars = handle('geom').variables()
        
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
        p_uv = figure(width=width, height=height,
                      tools='save,reset', toolbar_location='right')
        p_uv.add_tools(WheelZoomTool(),
                       BoxZoomTool(match_aspect=True))
        common_props(p_uv, xlim=(-20,20), ylim=(-20,20))
        p_uv.hex_tile(q=tc_vars['u'], r=tc_vars['vs'], source=vsrc, view=view,
                      size=1, fill_color='color', line_color='black', line_width=1, alpha=1.)    
        p_uv.add_tools(HoverTool(tooltips=[('u/v', '@'+tc_vars['u']+'/'+'@'+tc_vars['v']),]))
        
        ####### (x,y) plots ################################################################
        p_xy = figure(width=width, height=height,
                    tools='save,reset', toolbar_location='right',
                    output_backend='webgl')
        p_xy.add_tools(BoxZoomTool(match_aspect=True))
        p_xy.add_tools(HoverTool(tooltips=[('u/v', '@'+tc_vars['u']+'/'+'@'+tc_vars['v']),],))       
        common_props(p_xy, xlim=(-13,13), ylim=(-13,13))
        p_xy.rect(x=tc_vars['u'], y=tc_vars['v'], source=vsrc, view=view,
                  width=1., height=1., width_units='data', height_units='data',
                  fill_color='color', line_color='black',)
    
        ####### x vs. z plots ################################################################
        p_xVSz = figure(width=width, height=height, tools='save,reset', toolbar_location='right')
        p_xVSz.add_tools(BoxZoomTool(match_aspect=True))
        p_xVSz.scatter(x=tc_vars['z'], y=tc_vars['x'], source=vsrc)
        common_props(p_xVSz)
        
        ####### y vs. z plots ################################################################
        p_yVSz = figure(width=width, height=height, tools='save,reset', toolbar_location='right')
        p_yVSz.add_tools(BoxZoomTool(match_aspect=True))
        p_yVSz.scatter(x=tc_vars['z'], y=tc_vars['y'], source=vsrc)
        common_props(p_yVSz)
        
        ####### y vs. x plots ################################################################
        p_yVSx = figure(width=width, height=height, tools='save,reset', toolbar_location='right')
        p_yVSx.add_tools(BoxZoomTool(match_aspect=True))
        p_yVSx.scatter(x=tc_vars['x'], y=tc_vars['y'], source=vsrc)
        common_props(p_yVSx)
        
        ####### text input ###################################################################
        elements[ksrc]['textinput'].on_change('value', partial(text_callback, particle=ksrc, source=vsrc))

        ####### define layout ################################################################
        blank1 = Div(width=1000, height=100, text='')
        blank2 = Div(width=70, height=100, text='')

        lay = layout([[elements[ksrc]['textinput'], blank2, slider],
                      [p_uv,p_xy],
                      [blank1],
                      [p_xVSz,p_yVSz,p_yVSx]],
                      toolbar_options={'logo': None})

        tab = TabPanel(child=lay, title=ksrc)
        tabs.append(tab)
        # end for loop

    doc.add_root(Tabs(tabs=tabs))
    
parser = argparse.ArgumentParser(description='')
FLAGS = parser.parse_args()
display()
