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
pd.options.mode.chained_assignment = None # disable annoying SettingCopyWarning
import yaml
import logging
log = logging.getLogger(__name__)

from bokeh.plotting import figure, curdoc
from bokeh.util.hex import axial_to_cartesian
from bokeh import models as bmd
from bokeh import events as bev
from bokeh.palettes import viridis as _palette
mypalette = _palette(50)
from bokeh.layouts import layout
from bokeh.settings import settings
settings.ico_path = 'none'

import utils
from utils import params, common, parsing
import data_handle
from data_handle.data_handle import EventDataParticle
from data_handle.geometry import GeometryData

with open(params.viz_kw['CfgProdPath'], 'r') as afile:
    cfg_prod = yaml.safe_load(afile)
with open(params.viz_kw['CfgDataPath'], 'r') as afile:
    cfg_data = yaml.safe_load(afile)

data_part_opt = dict(tag='v2', reprocess=False, debug=True, logger=log)
data_particle = {
    'photons': EventDataParticle(particles='photons', **data_part_opt),
    'electrons': EventDataParticle(particles='electrons', **data_part_opt)}
geom_data = GeometryData(inname='test_triggergeom.root',
                         reprocess=False, logger=log)
mode = 'ev'

def common_props(p):
    p.output_backend = 'svg'
    p.toolbar.logo = None
    p.grid.visible = False
    p.outline_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

def get_data(event, particles):
    ds_geom = geom_data.provide()

    if mode=='ev':
        ds_ev = data_particle[particles].provide_event(event)
        ds_ev.rename(columns={'good_tc_waferu':'waferu', 'good_tc_waferv':'waferv',
                              'good_tc_cellu':'triggercellu', 'good_tc_cellv':'triggercellv',
                              'good_tc_layer':'layer'},
                    inplace=True)
        ds_ev = pd.merge(left=ds_ev, right=ds_geom, how='inner',
                         on=['layer', 'waferu', 'waferv', 'triggercellu', 'triggercellv'])
        return {'ev': ds_ev, 'geom': ds_geom}

    else:
        return {'geom': ds_geom}

if mode=='ev':
    def_evs = cfg_data['defaultEvents']
    def_ev_text = {}
    for k in def_evs:
        drop_text = [(str(q),str(q)) for q in def_evs[k]]
        def_ev_text[k] = drop_text

elements, cds_data = ({} for _ in range(2))
for k in (('photons', 'electrons') if mode=='ev' else ('Geometry',)):
    evs = def_evs[k][0] if mode == 'ev' else ''
    cds_data[k] = get_data(evs, k)[mode]
    elements[k] = {'source': bmd.ColumnDataSource(data=cds_data[k])}
    if mode=='ev':
        elements[k].update({'textinput': bmd.TextInput(placeholder='specify an event', height=40,
                                                       sizing_mode='stretch_width'),
                            'dropdown': bmd.Dropdown(label='Default Events', button_type='primary',
                                                    menu=def_ev_text[k], height=40)})

def text_callback(attr, old, new, source, particles):
    print('text callback ', particles, new)
    if not new.isdecimal():
        print('Wrong format!')
    else:
        source.data = get_data(int(new), particles)[mode]

def dropdown_callback(event, source, particles):
    print('dropdown callback', particles, int(event.__dict__['item']))
    source.data = get_data(int(event.__dict__['item']), particles)[mode]

def display():
    doc = curdoc()
    doc.title = 'HGCal Visualization'
    
    width, height   = 600, 600
    width2, height2 = 300, 200
    tabs = []

    vev = cfg_data['varEvents']
    
    for ksrc,vsrc in [(k,v['source']) for k,v in elements.items()]:

        if mode == 'ev':
            mapper_diams = bmd.LinearColorMapper(palette=mypalette,
                                                 low=vsrc.data[vev['en']].min(), high=vsrc.data[vev['en']].max())
            mapper_mods = bmd.LinearColorMapper(palette=mypalette,
                                                low=vsrc.data[vev['en']].min(), high=vsrc.data[vev['en']].max())  #CHANGE!!!!!!

        sld_opt = dict(bar_color='red', width=width, background='white')
        sld_layers = bmd.Slider(start=vsrc.data['layer'].min(), end=vsrc.data['layer'].max(),
                                value=vsrc.data['layer'].min(), step=2, title='Layer', **sld_opt)
        sld_layers_cb = bmd.CustomJS(args=dict(s=vsrc), code="""s.change.emit();""")
        sld_layers.js_on_change('value', sld_layers_cb) #value_throttled
        
        filt_layers = bmd.CustomJSFilter(args=dict(slider=sld_layers), code="""
           var indices = new Array(source.get_length());
           var sval = slider.value;
    
           const subset = source.data['layer'];
           for (var i=0; i < source.get_length(); i++) {
               indices[i] = subset[i] == sval;
           }
           return indices;
           """)

        if mode == 'ev':
            sld_en = bmd.Slider(start=cfg_prod['mipThreshold'], end=5,
                                value=cfg_prod['mipThreshold'], step=0.1,
                                title='Energy threshold [mip]', **sld_opt)
            sld_en_cb = bmd.CustomJS(args=dict(s=vsrc), code="""s.change.emit();""")
            sld_en.js_on_change('value', sld_en_cb) #value_throttled

            filt_en = bmd.CustomJSFilter(args=dict(slider=sld_en), code="""
               var indices = new Array(source.get_length());
               var sval = slider.value;
        
               const subset = source.data['good_tc_mipPt'];
               for (var i=0; i < source.get_length(); i++) {
                   indices[i] = subset[i] >= sval;
               }
               return indices;
               """)

        if mode == 'ev':
            view_cells = bmd.CDSView(filter=filt_layers & filt_en)
        else:
            view_cells = bmd.CDSView(filter=filt_layers)
        # modules are duplicated for cells lying in the same wafer
        # we want to avoid drawing the same module multiple times
        view_modules = (~cds_data[ksrc].duplicated(subset=['layer', 'waferu', 'waferv'])).tolist()
        #view_modules = bmd.CDSView(filter=filt_layers & bmd.BooleanFilter(view_modules))
        view_modules = bmd.CDSView(filter=filt_layers)

        ####### (u,v) plots ################################################################
        # p_uv = figure(width=width, height=height,
        #               tools='save,reset', toolbar_location='right')
        # p_uv.add_tools(bmd.WheelZoomTool(),
        #                bmd.BoxZoomTool(match_aspect=True))
        # common_props(p_uv, xlim=(-20,20), ylim=(-20,20))
        # p_uv.hex_tile(q=variables['tcwu'], r=variables['tcwv'], source=vsrc, view=view,
        #               size=1, fill_color='color', line_color='black', line_width=1, alpha=1.)    
        # p_uv.add_tools(bmd.HoverTool(tooltips=[('u/v', '@'+variables['tcwu']+'/'+'@'+variables['tcwv']),]))

        # find dataset minima and maxima
        cur_xmax, cur_ymax = -1e9, -1e9
        cur_xmin, cur_ymin = 1e9, 1e9
        for ex,ey in zip(vsrc.data['diamond_x'],vsrc.data['diamond_y']):
            if max(ex[0][0]) > cur_xmax: cur_xmax = max(ex[0][0])
            if min(ex[0][0]) < cur_xmin: cur_xmin = min(ex[0][0])
            if max(ey[0][0]) > cur_ymax: cur_ymax = max(ey[0][0])
            if min(ey[0][0]) < cur_ymin: cur_ymin = min(ey[0][0])
        # force matching ratio to avoid distortions
        distx, disty = cur_xmax-cur_xmin, cur_ymax-cur_ymin
        if distx > disty:
            cur_ymax += abs(distx-disty)/2
            cur_ymin = cur_ymax - distx
        else:
            cur_xmin -= abs(distx-disty)/2
            cur_xmax = cur_xmin + disty
        cur_xmax += (cur_xmax-cur_xmin)*0.05
        cur_xmin -= (cur_xmax-cur_xmin)*0.05
        cur_ymax += (cur_ymax-cur_ymin)*0.05
        cur_ymin -= (cur_ymax-cur_ymin)*0.05

        fig_opt = dict(width=width, height=height,
                       tools='save,reset,undo',
                       toolbar_location='right', output_backend='webgl'
                       )
        p_diams = figure(
            x_range=bmd.Range1d(cur_xmin, cur_xmax), y_range=bmd.Range1d(cur_ymin, cur_ymax),
            **fig_opt)
        p_mods = figure(x_range=p_diams.x_range, y_range=p_diams.y_range, **fig_opt)

        if mode == 'ev':
            hover_key_cells = 'Energy (cu,cv / wu,wv)'
            hover_val_cells = '@good_tc_mipPt (@triggercellu,@triggercellv / @waferu,@waferv)'
            hover_key_mods = 'Energy (wu,wv)'
            hover_val_mods = '@good_tc_mipPt (@waferu,@waferv)'
        else:
            hover_key_cells = 'cu,cv / wu,wv'
            hover_val_cells = '@triggercellu,@triggercellv / @waferu,@waferv'
            hover_key_mods = 'wu,wv'
            #hover_val_mods = '@waferu{custom},@waferv{custom}'
            hover_val_mods = '@waferu,@waferv'

        hover_code = """
        var wcoord = special_vars.{};
        return wcoord[0];
        """
        
        tool_list = (bmd.BoxZoomTool(match_aspect=True),)
        p_diams.add_tools(bmd.HoverTool(tooltips=[(hover_key_cells, hover_val_cells),],), *tool_list)
        # p_mods.add_tools(bmd.HoverTool(tooltips=[(hover_key_mods, hover_val_mods),],), *tool_list)
        #                                # formatters={'@waferu': bmd.CustomJSHover(code=hover_code.format('waferu')),
        #                                #             '@waferv': bmd.CustomJSHover(code=hover_code.format('waferv'))}

        p_mods.add_tools(*tool_list)
        common_props(p_diams)
        common_props(p_mods)

        polyg_opt = dict(line_color='black', line_width=2)
        p_diams_opt = dict(xs='diamond_x', ys='diamond_y', source=vsrc, view=view_cells, **polyg_opt)
        p_mods_opt = dict(xs='hex_x', ys='hex_y', source=vsrc, view=view_modules, **polyg_opt)
        hover_opt = dict(hover_fill_color='black', hover_line_color='black', hover_line_width=4, hover_alpha=0.2)

        if mode == 'ev':
            p_diams.multi_polygons(fill_color={'field': 'good_tc_mipPt', 'transform': mapper_diams},
                                   **hover_opt, **p_diams_opt)
            p_mods.multi_polygons(fill_color={'field': 'good_tc_mipPt', 'transform': mapper_mods}, #CHANGE WHEN MODULE SUMS ARE AVAILABLE
                                   **hover_opt, **p_mods_opt)

        else:
            p_diams.multi_polygons(color='green', **hover_opt, **p_diams_opt)
            #p_diams.circle(x='tc_x', y='tc_y', source=vsrc, view=view_cells, size=4, color='red', alpha=0.4)
            #p_diams.circle(x='x', y='y', source=vsrc, view=view_cells, size=10, color='blue')

            p_mods.multi_polygons(color='green', **hover_opt, **p_mods_opt)
                        
        if mode == 'ev':
            cbar_opt = dict(ticker=bmd.BasicTicker(desired_num_ticks=int(len(mypalette)/4)),
                            formatter=bmd.PrintfTickFormatter(format="%d"))
            cbar_diams = bmd.ColorBar(color_mapper=mapper_diams, title='TC energy [mipPt]', **cbar_opt)
            cbar_mods = bmd.ColorBar(color_mapper=mapper_mods, title='Module Sums [mipPt]', **cbar_opt)

            p_diams.add_layout(cbar_diams, 'right')
            p_mods.add_layout(cbar_mods, 'right')

            elements[ksrc]['textinput'].on_change('value', partial(text_callback, source=vsrc, particles=ksrc))
            elements[ksrc]['dropdown'].on_event('menu_item_click', partial(dropdown_callback, source=vsrc, particles=ksrc))

        ####### (x,y) plots ################################################################
        # p_xy = figure(width=width, height=height,
        #             tools='save,reset', toolbar_location='right',
        #             output_backend='webgl')
        # p_xy.add_tools(bmd.WheelZoomTool(), bmd.BoxZoomTool(match_aspect=True))
        # p_xy.add_tools(bmd.HoverTool(tooltips=[('u/v', '@'+variables['tcwu']+'/'+'@'+variables['tcwv']),],))       
        # common_props(p_xy, xlim=(-13,13), ylim=(-13,13))
        # p_xy.rect(x=variables['tcwu'], y=variables['tcwv'], source=vsrc, view=view,
        #           width=1., height=1., width_units='data', height_units='data',
        #           fill_color='color', line_color='black',)

        # ####### x vs. z plots ################################################################
        # p_xVSz = figure(width=width2, height=height2, tools='save,reset', toolbar_location='right')
        # p_xVSz.add_tools(bmd.BoxZoomTool(match_aspect=True))
        # p_xVSz.scatter(x=variables['z'], y=variables['x'], source=vsrc)
        # common_props(p_xVSz)
        
        # ####### y vs. z plots ################################################################
        # p_yVSz = figure(width=width2, height=height2, tools='save,reset', toolbar_location='right')
        # p_yVSz.add_tools(bmd.BoxZoomTool(match_aspect=True))
        # p_yVSz.scatter(x=variables['z'], y=variables['y'], source=vsrc)
        # common_props(p_yVSz)
        
        # ####### y vs. x plots ################################################################
        # p_yVSx = figure(width=width2, height=height2, tools='save,reset', toolbar_location='right')
        # p_yVSx.add_tools(bmd.BoxZoomTool(match_aspect=True))
        # p_yVSx.scatter(x=variables['x'], y=variables['y'], source=vsrc)
        # common_props(p_yVSx)
        
        ####### define layout ################################################################
        blank1 = bmd.Div(width=1000, height=100, text='')
        blank2 = bmd.Div(width=70, height=100, text='')

        if mode == 'ev':
            first_row = [elements[ksrc]['dropdown'], elements[ksrc]['textinput']]
        else:
            first_row = [sld_layers]
            
        lay = layout([first_row,
                      sld_layers,
                      sld_en,
                      #[p_diams, p_uv, p_xy],
                      #[p_xVSz, p_yVSz, p_yVSx],
                      [p_diams, p_mods],
                      [blank1],
                      ])
        tab = bmd.TabPanel(child=lay, title=ksrc)
        tabs.append(tab)
        # end for loop

    doc.add_root(bmd.Tabs(tabs=tabs))
    
parser = argparse.ArgumentParser(description='')
FLAGS = parser.parse_args()
logging.basicConfig()
display()
