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

with open(params.CfgPaths['prod'], 'r') as afile:
    cfg_prod = yaml.safe_load(afile)
with open(params.CfgPaths['data'], 'r') as afile:
    cfg_data = yaml.safe_load(afile)

data_part_opt = dict(tag='mytag', reprocess=False, debug=True, logger=log)
data_particle = {
    'photons': EventDataParticle(particles='photons', **data_part_opt),
    #'electrons': EventDataParticle(particles='electrons', **data_part_opt)
}
geom_data = GeometryData(reprocess=True, logger=log)
mode = 'geom'

def common_props(p):
    p.output_backend = 'svg'
    p.toolbar.logo = None
    p.grid.visible = False
    p.outline_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

def get_data(event, particles):
    region, section = None, 'si'
    if mode == event:
        assert region is None
    ds_geom = geom_data.provide(section=section, region=region)

    if mode=='ev':
        tc_keep = {'good_tc_waferu'     : 'waferu',
                   'good_tc_waferv'     : 'waferv',
                   'good_tc_cellu'      : 'triggercellu',
                   'good_tc_cellv'      : 'triggercellv',
                   'good_tc_layer'      : 'layer',
                   'good_tc_pt'         : 'tc_pt',
                   'good_tc_mipPt'      : 'tc_mipPt',
                   'good_tc_cluster_id' : 'tc_cluster_id'}

        ds_ev = data_particle[particles].provide_event(event)
        ds_ev = ds_ev['tc']
        ds_ev = ds_ev.rename(columns=tc_keep)
        ds_ev = ds_ev[tc_keep.values()]

        ds_ev = pd.merge(left=ds_ev, right=ds_geom, how='inner',
                         on=['layer', 'waferu', 'waferv',
                             'triggercellu', 'triggercellv'])
        return {'ev': ds_ev, 'geom': ds_geom}

    else:
        return ds_geom

if mode=='ev':
    def_evs = cfg_data['defaultEvents']
    def_ev_text = {}
    for k in def_evs:
        drop_text = [(str(q),str(q)) for q in def_evs[k]]
        def_ev_text[k] = drop_text

elements = {}
if mode=='ev':
    evsource = {}
    for k in data_particle.keys():
        evs = def_evs[k][0]
        evdata = get_data(evs, k)[mode]
        evsource[k] = {'si': bmd.ColumnDataSource(data=evdata['si']),
                       'sci': bmd.ColumnDataSource(data=evdata['sci'])}
        elements[k].update({'textinput': bmd.TextInput(placeholder=str(def_evs[k][0]), height=40,
                                                       sizing_mode='stretch_width'),
                            'dropdown': bmd.Dropdown(label='Default Events', button_type='primary',
                                                    menu=def_ev_text[k], height=40)})
else:
    geomdata = get_data('', 'Geometry')
    gsource = {'si': bmd.ColumnDataSource(data=geomdata['si']),
               'sci': bmd.ColumnDataSource(data=geomdata['sci'])}
    
    
def range_callback(fig, source, xvar, yvar):
    """18 (centimeters) makes sures entire modules are always within the area shown"""
    fig.x_range.start = min(source.data[xvar]-18)
    fig.x_range.end = max(source.data[xvar]+18)
    fig.y_range.start = min(source.data[yvar]-18)
    fig.y_range.end = max(source.data[yvar]+18)

def text_callback(attr, old, new, source, figs, particles):
    print('text callback ', particles, new)
    if not new.isdecimal():
        print('Wrong format!')
    else:
        source.data = get_data(int(new), particles)[mode]
    for fig in figs:
        range_callback(fig, source, 'tc_x', 'tc_y')

def dropdown_callback(event, source, figs, particles):
    print('dropdown callback', particles, int(event.__dict__['item']))
    source.data = get_data(int(event.__dict__['item']), particles)[mode]
    for fig in figs:
        range_callback(fig, source, 'tc_x', 'tc_y')

def display():
    doc = curdoc()
    doc.title = 'HGCal Visualization'
    
    width, height   = 600, 600
    width2, height2 = 300, 200
    ven, vl = 'tc_mipPt', 'layer'

    if mode == 'ev':
        for ksrc,vsrc in [(k,v) for k,v in evsource.items()]:

            mapper_diams = bmd.LinearColorMapper(palette=mypalette,
                                                 low=vsrc.data[ven].min(),
                                                 high=vsrc.data[ven].max())
            # mapper_mods = bmd.LinearColorMapper(palette=mypalette,
            #                                     low=vsrc.data[ven].min(),
            #                                     high=vsrc.data[ven].max())  #CHANGE!!!!!!
        sld_opt = dict(bar_color='red', width=width, background='white')
        sld_layers = bmd.Slider(start=vsrc.data[vl].min(), end=vsrc.data[vl].max(),
                                value=vsrc.data[vl].min(), step=2, title='Layer', **sld_opt)
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

        sld_en = bmd.Slider(start=0, end=5, step=0.1,
                            value=cfg_prod['selection']['mipThreshold'], 
                            title='Energy threshold [mip]', **sld_opt)
        sld_en_cb = bmd.CustomJS(args=dict(s=vsrc), code="""s.change.emit();""")
        sld_en.js_on_change('value', sld_en_cb) #value_throttled

        filt_en = bmd.CustomJSFilter(args=dict(slider=sld_en), code="""
           var indices = new Array(source.get_length());
           var sval = slider.value;
    
           const subset = source.data['tc_mipPt'];
           for (var i=0; i < source.get_length(); i++) {
               indices[i] = subset[i] >= sval;
           }
           return indices;
           """)

        all_filters = filt_layers & filt_en

        view_cells = bmd.CDSView(filter=all_filters)
        # modules are duplicated for cells lying in the same wafer
        # we want to avoid drawing the same module multiple times
        #view_modules = (~cds[ksrc].duplicated(subset=[vl, 'waferu', 'waferv'])).tolist()
        #view_modules = bmd.CDSView(filter=filt_layers & bmd.BooleanFilter(view_modules))
        view_modules = bmd.CDSView(filter=all_filters)

        # find dataset minima and maxima
        cur_xmax, cur_ymax = -1e9, -1e9
        cur_xmin, cur_ymin = 1e9, 1e9

        zobj = (vsrc.data['diamond_x'],vsrc.data['diamond_y'],vsrc.data[ven])

        for elem in zip(*zobj):
            # cut replicates the default `view_en`
            if mode == 'ev' and elem[2] < cfg_prod['selection']['mipThreshold']:
                continue
            if max(elem[0][0][0]) > cur_xmax: cur_xmax = max(elem[0][0][0])
            if min(elem[0][0][0]) < cur_xmin: cur_xmin = min(elem[0][0][0])
            if max(elem[1][0][0]) > cur_ymax: cur_ymax = max(elem[1][0][0])
            if min(elem[1][0][0]) < cur_ymin: cur_ymin = min(elem[1][0][0])
                    
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
                       x_axis_label='X [cm]', y_axis_label='Y [cm]',
                       tools='save,reset,undo',
                       toolbar_location='right', output_backend='webgl'
                       )
        # p_tc = figure(x_range=bmd.Range1d(cur_xmin, cur_xmax),
        #               y_range=bmd.Range1d(cur_ymin, cur_ymax), **fig_opt)
        p_tc = figure(match_aspect=True, **fig_opt)
        # p_mods = figure(
        #     #x_range=p_tc.x_range, y_range=p_tc.y_range,
        #     x_range=bmd.Range1d(-200, 200), y_range=bmd.Range1d(-200, 200),
        #     **fig_opt)

        hover_key_cells = 'Energy (cu,cv / wu,wv)'
        hover_val_cells = '@{} (@triggercellu,@triggercellv / @waferu,@waferv)'.format(ven)
        # hover_key_mods = 'Energy (wu,wv)'
        # hover_val_mods = '@{} (@waferu,@waferv)'.format(ven)
        hover_code = """
        var wcoord = special_vars.{};
        return wcoord[0];
        """
        
        tool_list = (bmd.BoxZoomTool(match_aspect=True),)
        p_tc.add_tools(bmd.HoverTool(tooltips=[(hover_key_cells, hover_val_cells),],), *tool_list)
        # p_mods.add_tools(bmd.HoverTool(tooltips=[(hover_key_mods, hover_val_mods),],), *tool_list)
        #                                # formatters={'@waferu': bmd.CustomJSHover(code=hover_code.format('waferu')),
        #                                #             '@waferv': bmd.CustomJSHover(code=hover_code.format('waferv'))}

        # p_mods.add_tools(*tool_list)
        common_props(p_tc)
        # common_props(p_mods)

        polyg_opt = dict(line_color='black', line_width=2)
        tc_opt = dict(xs='diamond_x', ys='diamond_y',
                      source=vsrc, view=view_cells, **polyg_opt)
        # mods_opt = dict(xs='hex_x', ys='hex_y',
        #                 source=vsrc, view=view_modules, **polyg_opt)
        hover_opt = dict(hover_fill_color='black', hover_line_color='black', hover_line_width=4, hover_alpha=0.2)

        p_tc.multi_polygons(fill_color={'field': ven, 'transform': mapper_diams},
                            **hover_opt, **tc_opt)
        # p_mods.multi_polygons(fill_color={'field': ven, 'transform': mapper_mods}, #CHANGE WHEN MODULE SUMS ARE AVAILABLE
        #                       **hover_opt, **mods_opt)
                        
        cbar_opt = dict(ticker=bmd.BasicTicker(desired_num_ticks=int(len(mypalette)/4)),
                        formatter=bmd.PrintfTickFormatter(format="%d"))
        cbar_diams = bmd.ColorBar(color_mapper=mapper_diams, title='TC energy [mipPt]', **cbar_opt)
        # cbar_mods = bmd.ColorBar(color_mapper=mapper_mods, title='Module Sums [mipPt]', **cbar_opt)

        p_tc.add_layout(cbar_diams, 'right')
        # p_mods.add_layout(cbar_mods, 'right')
        # p_tc.x_range.callback = bmd.CustomJS(args=dict(xrange=myplot.x_range), code="""
        # xrange.set({"start": 10, "end": 20})
        # """)

        # elements[ksrc]['textinput'].on_change('value', partial(text_callback, source=vsrc,
        #                                                        figs=(p_tc, p_mods), particles=ksrc))           
        # elements[ksrc]['dropdown'].on_event('menu_item_click', partial(dropdown_callback, source=vsrc,
        #                                                                    figs=(p_tc,p_mods), particles=ksrc))
        elements[ksrc]['textinput'].on_change('value', partial(text_callback, source=vsrc,
                                                               figs=(p_tc,), particles=ksrc))
        elements[ksrc]['dropdown'].on_event('menu_item_click', partial(dropdown_callback, source=vsrc,
                                                                       figs=(p_tc,), particles=ksrc))

    elif mode == 'geom':
        keymin = 'si' if len(gsource['si'].data[vl])!=0 else 'sci'
        keymax = 'sci' if len(gsource['sci'].data[vl])!=0 else 'si'
        sld_opt = dict(bar_color='red', width=width, background='white')
        sld_layers = bmd.Slider(start=gsource[keymin].data[vl].min(), end=gsource[keymax].data[vl].max(),
                                value=gsource[keymax].data[vl].min(), step=1, title='Layer', **sld_opt)
        sld_layers_cb = bmd.CustomJS(args=dict(s1=gsource['si'], s2=gsource['sci']),
                                     code="""s1.change.emit(); s2.change.emit();""")
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

        view_si = bmd.CDSView(filter=filt_layers)
        view_sci = bmd.CDSView(filter=filt_layers)
        # modules are duplicated for cells lying in the same wafer
        # we want to avoid drawing the same module multiple times
        #view_modules = (~cds[ksrc].duplicated(subset=[vl, 'waferu', 'waferv'])).tolist()
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
        xmax_si, ymax_si, xmax_sci, ymax_sci = (-1e9 for _ in range(4))
        xmin_si, ymin_si, xmin_sci, ymin_sci = (+1e9 for _ in range(4))
        zobj_si = gsource['si'].data['diamond_x'], gsource['si'].data['diamond_y']
        zobj_sci = gsource['sci'].data['rmax'], gsource['sci'].data['phimax']

        # scintillator minima and maxima
        if len(zobj_sci[0])>0:
            xproxy_sci = zobj_sci[0]*np.cos(zobj_sci[1])
            yproxy_sci = zobj_sci[0]*np.sin(zobj_sci[1])
            xmaxproxy_sci, ymaxproxy_sci = max(xproxy_sci), max(yproxy_sci)
            xminproxy_sci, yminproxy_sci = min(xproxy_sci), min(yproxy_sci)
            if xmaxproxy_sci > xmax_sci: xmax_sci = xmaxproxy_sci
            if xminproxy_sci < xmin_sci: xmin_sci = xminproxy_sci
            if ymaxproxy_sci > ymax_sci: ymax_sci = ymaxproxy_sci
            if yminproxy_sci < ymin_sci: ymin_sci = yminproxy_sci

        # silicon minima and maxima
        for elem in zip(*zobj_si):
            if max(elem[0][0][0]) > xmax_si: xmax_si = max(elem[0][0][0])
            if min(elem[0][0][0]) < xmin_si: xmin_si = min(elem[0][0][0])
            if max(elem[1][0][0]) > ymax_si: ymax_si = max(elem[1][0][0])
            if min(elem[1][0][0]) < ymin_si: ymin_si = min(elem[1][0][0])

        # global minima and maxima
        xmax, ymax = max(xmax_sci, xmax_si), max(ymax_sci, ymax_si)
        xmin, ymin = min(xmin_sci, xmin_si), min(ymin_sci, ymin_si)
        
        # force matching ratio to avoid distortions
        distx, disty = xmax-xmin, ymax-ymin
        if distx > disty:
            ymax += abs(distx-disty)/2
            ymin = ymax - distx
        else:
            xmin -= abs(distx-disty)/2
            xmax = xmin + disty
        xmax += (xmax-xmin)*0.05
        xmin -= (xmax-xmin)*0.05
        ymax += (ymax-ymin)*0.05
        ymin -= (ymax-ymin)*0.05

        fig_opt = dict(width=width, height=height,
                       x_axis_label='X [cm]', y_axis_label='Y [cm]',
                       tools='save,reset,undo',
                       toolbar_location='right', output_backend='webgl')

        # p_tc = figure(x_range=bmd.Range1d(xmin, xmax),
        #               y_range=bmd.Range1d(ymin, ymax),
        #               **fig_opt)
        p_tc = figure(match_aspect=True, **fig_opt)
        # p_mods = figure(x_range=bmd.Range1d(-200, 200), y_range=bmd.Range1d(-200, 200),
        #                 #x_range=p_tc.x_range, y_range=p_tc.y_range,
        #                 **fig_opt)

        # p_mods.add_tools(bmd.HoverTool(tooltips=[(hover_key_mods, hover_val_mods),],), *tool_list)
        #                                # formatters={'@waferu': bmd.CustomJSHover(code=hover_code.format('waferu')),
        #                                #             '@waferv': bmd.CustomJSHover(code=hover_code.format('waferv'))}
        # p_mods.add_tools(*tool_list)
        common_props(p_tc)
        # common_props(p_mods)

        polyg_opt = dict(line_color='black', line_width=2)
        # mods_opt = dict(xs='hex_x', ys='hex_y',
        #                 source=gsource['si'], view=view_modules, **polyg_opt)
        hover_opt = dict(hover_fill_color='black', hover_line_color='black', hover_line_width=4, hover_alpha=0.2)

        r_si = p_tc.multi_polygons(xs='diamond_x', ys='diamond_y',
                                   color='green', 
                                   source=gsource['si'], view=view_si,
                                   **polyg_opt, **hover_opt)
        r_sci = p_tc.annular_wedge(x=0., y=0.,
                                   inner_radius='rmin', outer_radius='rmax',
                                   start_angle='phimin', end_angle='phimax',
                                   inner_radius_units='data', outer_radius_units='data',
                                   start_angle_units='rad', end_angle_units='rad',
                                   source=gsource['sci'], view=view_sci,
                                   color='red', **polyg_opt)
        # r_xy = p_tc.circle(x='x', y='y', source=gsource['sci'], view=view_sci,
        #                    size=2., color='blue', legend_label='true')

        hvr_tt_si = [('cu,cv / wu,wv',
                      '@triggercellu,@triggercellv / @waferu,@waferv')]
        hvr_tt_sci = [('iphi,ieta',
                       '@triggercelliphi,@triggercellieta')]
        # hvr_tt_xy = [('iphi,ieta',
        #               '@triggercelliphi,@triggercellieta')]
        hvr_si = bmd.HoverTool(tooltips=hvr_tt_si, renderers=[r_si])
        hvr_sci = bmd.HoverTool(tooltips=hvr_tt_sci, renderers=[r_sci])
        # hvr_xy = bmd.HoverTool(tooltips=hvr_tt_xy, renderers=[r_xy])
        tool_list = (bmd.BoxZoomTool(match_aspect=True),)
        p_tc.add_tools(hvr_si, hvr_sci, *tool_list)

        p_tc.circle(x='tc_x', y='tc_y', source=gsource['si'], view=view_sci,
                    size=5, color='blue', legend_label='u,v conversion')
        p_tc.circle(x='x', y='y', source=gsource['si'], view=view_sci,
                    size=5, color='orange', legend_label='tc original')
        p_tc.legend.click_policy='hide'
        # p_mods.multi_polygons(color='green', **hover_opt, **mods_opt)
                        
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
        blank = bmd.Div(width=1000, height=100, text='')
        if mode == 'ev':
            tabs = []
            for ksrc,vsrc in [(k,v) for k,v in evsource.items()]:
                first_row = [elements[ksrc]['dropdown'], elements[ksrc]['textinput']]
                lay = layout([first_row,
                              sld_layers,
                              sld_en,
                              #[p_tc, p_uv, p_xy],
                              #[p_xVSz, p_yVSz, p_yVSx],
                              [p_tc],
                              [blank],
                              ])
                tab = bmd.TabPanel(child=lay, title=ksrc)
                tabs.append(tab)
            doc.add_root(bmd.Tabs(tabs=tabs))
        else:
            first_row = [sld_layers]            
            lay = layout([first_row,
                          #[p_tc, p_uv, p_xy],
                          #[p_xVSz, p_yVSz, p_yVSx],
                          [p_tc],
                          [blank],
                        ])
            doc.add_root(lay)
            
parser = argparse.ArgumentParser(description='')
FLAGS = parser.parse_args()
logging.basicConfig()
display()
