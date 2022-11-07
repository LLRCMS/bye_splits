# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import argparse
import numpy as np
import uproot as up

from bokeh.io import output_file, save
from bokeh.plotting import figure
from bokeh.util.hex import axial_to_cartesian
from bokeh.models import (
    Div,
    Panel,
    Tabs,
    BoxZoomTool,
    Range1d,
    ColumnDataSource,
    HoverTool,
    Slider,
    CustomJS,
    CustomJSFilter,
    CDSView,
    )
from bokeh.layouts import layout
output_file('tmp.html')

import utils
from utils import params, common, parsing
from surface_3d import Surface3d
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

def display(reprocess):
        width = int(1600/3)
        height = 400
         
        tc_data = handle('geom').provide(reprocess)
        tc_vars = handle('geom').variables()
         
        source = ColumnDataSource(tc_data)
        slider  = Slider(start=tc_data.layer.min(), end=tc_data.layer.max(),
                         value=tc_data.layer.min(), step=2, title='Layer',
                         bar_color='red', default_size=800,
                         background='white')
        callback = CustomJS(args=dict(s=source), code="""
            s.change.emit();
        """)
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
         
         
        ##########################
        # 3D plot
        # x3d = np.arange(0, 300, 10)
        # y3d = np.arange(0, 300, 10)
        # xx3d, yy3d = np.meshgrid(x3d, y3d)
        # xx3d = xx3d.ravel()
        # yy3d = yy3d.ravel()
        # value3d = np.sin(xx3d / 50) * np.cos(yy3d / 50) * 50 + 50
         
        # source3d = ColumnDataSource(data=dict(x=xx3d, y=yy3d, z=value3d))
         
        surface = Surface3d(x='z', y='y', z='x', data_source=source)
        ##########################
         
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
         
        blank = Div(width=1000, height=100, text='')
        lay = layout([[slider],[p_uv,p_xy,surface],[blank],[p_xVSz,p_yVSz,p_yVSx]])
        save(lay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--reprocess',
                        help='reprocess trigger cell geometry data',
                        action='store_true')
    FLAGS = parser.parse_args()
    
    display(FLAGS.reprocess)
