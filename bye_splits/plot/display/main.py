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
import yaml

#from bokeh.io import output_file, save
#output_file('tmp.html')
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

with open(params.viz_kw['CfgEventPath'], 'r') as afile:
    config = yaml.safe_load(afile)

data_part_opt = dict(tag='v2', reprocess=False, debug=True)
data_particle = {
    'photons': EventDataParticle(particles='photons', **data_part_opt),
    'electrons': EventDataParticle(particles='electrons', **data_part_opt)}
geom_data = GeometryData(inname='test_triggergeom_v2.root', reprocess=True)
data_vars = {'ev': config['varEvents'],
             'geom': config['varGeometry']}
mode = 'geom'

def common_props(p, xlim=None, ylim=None):
    p.output_backend = 'svg'
    p.toolbar.logo = None
    p.grid.visible = False
    p.outline_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False
    if xlim is not None:
        p.x_range = bmd.Range1d(xlim[0], xlim[1])
    if ylim is not None:
        p.y_range = bmd.Range1d(ylim[0], ylim[1])
        
def convert_cells_to_xy(df):
    scu, scv = 'triggercellu', 'triggercellv'
    swu, swv = 'waferu', 'waferv'
    
    c30, s30, t30 = np.sqrt(3)/2, 1/2, 1/np.sqrt(3)
    N = 4
    N2 = 2*N - 1
    waferSize = 1
    cellDistX = waferSize/8.
    cellDistY = cellDistX * t30
    R = waferSize / (3 * N)
    r = R * c30

    cells_conversion = {
        0: (lambda cu,cv: (1.5*(cu-cv)+0.5) * R,
            lambda cu,cv: (cv+cu-2*N+1) * r),
        1: (lambda cu,cv: (1.5*(cv-N)+0.5) * R,
            lambda cu,cv: -(2*cv-cu-N+1) * r),
        2: (lambda cu,cv: -(1.5*(cu-N)+1) * R,
            lambda cu,cv: -(2*cv-cu-N) * r),
        3: (lambda cu,cv: -(1.5*(cu-cv)+0.5) * R,
            lambda cu,cv: -(cv+cu-2*N+1) * r),
        4: (lambda cu,cv: (1.5*(cu-N)+0.5) * R,
            lambda cu,cv: -(2*cu-cv-N+1) * r),
        5: (lambda cu,cv: (1.5*(cu-N)+1) * R,
            lambda cu,cv: (2*cv-cu-N) * r),
        6: (lambda cu,cv: (1.5*(cv-cu)+0.5) * R,
            lambda cu,cv: (cv+cu-2*N+1) * r),
        7: (lambda cu,cv: (1.5*(cv-N)+1) * R,
            lambda cu,cv: (2*cu-cv-N) * r),
        8: (lambda cu,cv: (1.5*(cu-N)+0.5) * R,
            lambda cu,cv: -(2*cv-cu-N+1) * r),
        9: (lambda cu,cv: -(1.5*(cv-cu)+0.5) * R,
            lambda cu,cv: -(cv+cu-2*N+1) * r),
        10: (lambda cu,cv: -(1.5*(cv-N)+1) * R,
             lambda cu,cv: -(2*cu-cv-N) * r),
        11: (lambda cu,cv: -(1.5*(cu-N)+0.5) * R,
             lambda cu,cv: (2*cv-cu-N+1) * r),
    }
    wafer_shifts = (lambda wu,wv,cx: (2*wu - wv)*waferSize/2 + cx,
                    lambda wv,cy: c30*wv + cy)

    # https://indico.cern.ch/event/1111846/contributions/4675222/attachments/2373114/4053810/v17.pdf
    masks_index_placement = lambda ori : df['waferorient'] + 6 == ori
    # masks_location = {'UL': (df[scu]>=N) & (df[scu]<=N2) & (df[scv]<df[scu]),
    #                   'UR': (df[scv]>=N) & (df[scv]<=N2) & (df[scv]>=df[scu]),
    #                   'B':  (df[scu]<N) & (df[scv]<N),
    #                   } #up-right, up-left and bottom
    def masks_location(location, ax, ay): #up-right, up-left and bottom
        if location == 'UL':
            return df[ax] < 0 and df[ay] > 0
        elif location == 'UR':
            return df[ax] > 0 and df[ay] > 0
        else:
            return df[ay] < 0

    x0, x1, x2, x3 = ({} for _ in range(4))
    y0, y1, y2, y3 = ({} for _ in range(4))
    xaxis, yaxis = ({} for _ in range(2))
    xaxis_plac, yaxis_plac = ({} for _ in range(2))

    for ip_key in range(6,12): #orientation indices
        xaxis_plac.update({ip_key: {}})
        yaxis_plac.update({ip_key: {}})
        masks_ip = masks_index_placement(ip_key)

        cu_data = df[scu][masks_ip]
        cv_data = df[scv][masks_ip]
        wu_data = df[swu][masks_ip]
        wv_data = df[swv][masks_ip]

        cx_data = cells_conversion[ip_key][0](cu_data, cv_data)
        cy_data = cells_conversion[ip_key][1](cu_data, cv_data)
        wx_data = wafer_shifts[0](wu_data, wv_data, cx_data)
        wy_data = wafer_shifts[1](wv_data, cy_data)

        for loc_key in ('UL', 'UR', 'B'):
            masks_loc = masks_location(loc_key, cx_data, cy_data)
            breakpoint()
            wx_d, wy_d = wx_data[masks_loc], wy_data[masks_loc]
            
            # x0 refers to the x position the lefmost, down corner all diamonds (TCs)
            # x1, x2, x3 are defined in a counter clockwise fashion
            # same for y0, y1, y2 and y3
            # tc positions refer to the center of the diamonds
            if loc_key in ('UL', 'UR'):
                x0.update({loc_key: wx_d})
            else:
                x0.update({loc_key: wx_d - cellDistX})
            x1.update({loc_key: x0[loc_key][:] + cellDistX})
            if loc_key in ('UL', 'UR'):
                x2.update({loc_key: x1[loc_key]})
                x3.update({loc_key: x0[loc_key]})
            else:
                x2.update({loc_key: x1[loc_key] + cellDistX})
                x3.update({loc_key: x1[loc_key]})

            if loc_key in ('UL', 'UR'):
                y0.update({loc_key: wy_d})
            else:
                y0.update({loc_key: wy_d + cellDistY})
            if loc_key in ('UR', 'B'):
                y1.update({loc_key: y0[loc_key][:] - cellDistY})
            else:
                y1.update({loc_key: y0[loc_key][:] + cellDistY})
            if loc_key in ('B'):
                y2.update({loc_key: y0[loc_key][:]})
            else:
                y2.update({loc_key: y1[loc_key][:] + 2*cellDistY})
            if loc_key in ('UL', 'UR'):
                y3.update({loc_key: y0[loc_key][:] + 2*cellDistY})
            else:
                y3.update({loc_key: y0[loc_key][:] + cellDistY})

            keys = ['pos0','pos1','pos2','pos3']
            xaxis.update({
                loc_key: pd.concat([x0[loc_key],x1[loc_key],x2[loc_key],x3[loc_key]],
                                   axis=1, keys=keys)})
            yaxis.update(
                {loc_key: pd.concat([y0[loc_key],y1[loc_key],y2[loc_key],y3[loc_key]],
                                    axis=1, keys=keys)})

            xaxis[loc_key]['new'] = [[round(val, 3) for val in sublst]
                                     for sublst in xaxis[loc_key].values.tolist()]
            yaxis[loc_key]['new'] = [[round(val, 3) for val in sublst]
                                     for sublst in yaxis[loc_key].values.tolist()]
            # xaxis[loc_key]['x_tmp'] = x0[loc_key]
            # yaxis[loc_key]['y_tmp'] = y0[loc_key]

            xaxis[loc_key] = xaxis[loc_key].drop(keys, axis=1)
            yaxis[loc_key] = yaxis[loc_key].drop(keys, axis=1)

        xaxis_plac[ip_key] = pd.concat(xaxis.values())
        yaxis_plac[ip_key] = pd.concat(yaxis.values())
        xaxis_plac[ip_key] = xaxis_plac[ip_key].groupby(xaxis_plac[ip_key].index).agg(lambda k: [k])
        yaxis_plac[ip_key] = yaxis_plac[ip_key].groupby(yaxis_plac[ip_key].index).agg(lambda k: [k])

    tc_polyg_x = pd.concat(xaxis_plac.values())
    tc_polyg_y = pd.concat(yaxis_plac.values())
    res = pd.concat([tc_polyg_x, tc_polyg_y], axis=1)
    #res.columns = ['tc_polyg_x', 'x_tmp', 'tc_polyg_y', 'y_tmp']
    res.columns = ['tc_polyg_x', 'tc_polyg_y']
    return df.join(res)

def get_data(event, particles):
    ds_geom = geom_data.provide()
    # ds_geom = ds_geom[((ds_geom[data_vars['geom']['wu']]==3) & (ds_geom[data_vars['geom']['wv']]==3)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==3) & (ds_geom[data_vars['geom']['wv']]==4)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==4) & (ds_geom[data_vars['geom']['wv']]==3)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==4) & (ds_geom[data_vars['geom']['wv']]==4))]
    ds_geom = ds_geom[((ds_geom[data_vars['geom']['wu']]==-7) & (ds_geom[data_vars['geom']['wv']]==3)) |
                      ((ds_geom[data_vars['geom']['wu']]==-8) & (ds_geom[data_vars['geom']['wv']]==2)) |
                      ((ds_geom[data_vars['geom']['wu']]==-8) & (ds_geom[data_vars['geom']['wv']]==1)) |
                      ((ds_geom[data_vars['geom']['wu']]==-7) & (ds_geom[data_vars['geom']['wv']]==2))]
    ds_geom = convert_cells_to_xy(ds_geom)

    ds_ev = data_particle[particles].provide_event(event)
    ds_ev.rename(columns={'good_tc_waferu':'waferu', 'good_tc_waferv':'waferv',
                          'good_tc_cellu':'triggercellu', 'good_tc_cellv':'triggercellv',
                          'good_tc_layer':'layer'},
                 inplace=True)

    ds_ev = pd.merge(left=ds_ev, right=ds_geom, how='inner',
                     on=['layer', 'waferu', 'waferv', 'triggercellu', 'triggercellv'])
    #ds_ev = convert_cells_to_xy(ds_ev)

    return {'ev': ds_ev, 'geom': ds_geom}

with open(params.viz_kw['CfgEventPath'], 'r') as afile:
    cfg = yaml.safe_load(afile)
    def_evs = cfg['defaultEvents']
    def_ev_text = {}
    for k in def_evs:
        drop_text = [(str(q),str(q)) for q in def_evs[k]]
        def_ev_text[k] = drop_text

elements = {}
for k in ('photons', 'electrons'):
    elements[k] = {'textinput': bmd.TextInput(value='<specify an event>', height=40,
                                             sizing_mode='stretch_width'),
                   'dropdown': bmd.Dropdown(label='Default Events', button_type='primary',
                                            menu=def_ev_text[k], height=40,),
                   'source': bmd.ColumnDataSource(data=get_data(def_evs[k][0], k)[mode])}

def text_callback(attr, old, new, source, particles):
    print('running ', particles, new)
    if not new.isdecimal():
        print('Wrong format!')
    else:
        source.data = get_data(int(new), particles)[mode]

def dropdown_callback(event, source, particles):
    source.data = get_data(int(event.__dict__['item']), particles)[mode]

def display():
    doc = curdoc()
    doc.title = 'TC Visualization'
    
    width, height   = 1250, 1000
    width2, height2 = 400, 300
    tabs = []
    
    for ksrc,vsrc in [(k,v['source']) for k,v in elements.items()]:
        if mode == 'ev':
            mapper = bmd.LinearColorMapper(palette=mypalette,
                                          low=vsrc.data['good_tc_mipPt'].min(), high=vsrc.data['good_tc_mipPt'].min())

        slider = bmd.Slider(start=vsrc.data['layer'].min(), end=vsrc.data['layer'].max(),
                            value=vsrc.data['layer'].min(), step=2, title='Layer',
                            bar_color='red', width=600, background='white')
        slider_callback = bmd.CustomJS(args=dict(s=vsrc), code="""s.change.emit();""")
        slider.js_on_change('value', slider_callback) #value_throttled

        view = bmd.CDSView(filter=bmd.CustomJSFilter(args=dict(slider=slider), code="""
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
        # p_uv.add_tools(bmd.WheelZoomTool(),
        #                bmd.BoxZoomTool(match_aspect=True))
        # common_props(p_uv, xlim=(-20,20), ylim=(-20,20))
        # p_uv.hex_tile(q=variables['tcwu'], r=variables['tcwv'], source=vsrc, view=view,
        #               size=1, fill_color='color', line_color='black', line_width=1, alpha=1.)    
        # p_uv.add_tools(bmd.HoverTool(tooltips=[('u/v', '@'+variables['tcwu']+'/'+'@'+variables['tcwv']),]))

        ####### cell plots ################################################################
        lim = 22
        polyg_opt = dict(line_color='black', line_width=2)
        p_cells = figure(width=width, height=height,
                         x_range=bmd.Range1d(-lim, lim),
                         y_range=bmd.Range1d(-lim, lim),
                         tools='save,reset', toolbar_location='right',
                         output_backend='webgl')

        hover_val_common = '@triggercellu,@triggercellv / @waferu,@waferv'
        if mode == 'ev':
            hover_key = 'Energy (cu,cv / wu,wv)'
            hover_val = '@good_tc_mipPt (' + hover_val_common + ')'
        else:
            hover_key = 'cu,cv / wu,wv'
            hover_val = hover_val_common

        p_cells.add_tools(bmd.BoxZoomTool(match_aspect=True),
                          bmd.WheelZoomTool(),
                          bmd.HoverTool(tooltips=[(hover_key, hover_val),]))
        common_props(p_cells, xlim=(-lim, lim), ylim=(-lim, lim))

        p_cells_opt = dict(xs='tc_polyg_x', ys='tc_polyg_y', source=vsrc, view=view, **polyg_opt)

        if mode == 'ev':
            p_cells.multi_polygons(fill_color={'field': 'good_tc_mipPt', 'transform': mapper},
                                   **p_cells_opt)
        else:
            p_cells.multi_polygons(color='green', **p_cells_opt)
            #p_cells.circle(x='x_tmp', y='y_tmp', source=vsrc, size=3)
                        
        if mode == 'ev':
            color_bar = bmd.ColorBar(color_mapper=mapper,
                                     ticker=bmd.BasicTicker(desired_num_ticks=int(len(mypalette)/4)),
                                     formatter=bmd.PrintfTickFormatter(format="%d"))
            p_cells.add_layout(color_bar, 'right')

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
        
        ####### text input ###################################################################
        elements[ksrc]['textinput'].on_change('value', partial(text_callback, source=vsrc, particles=ksrc))

        slider_callback = bmd.CustomJS(args=dict(s=vsrc), code="""s.change.emit();""")
        slider.js_on_change('value', slider_callback) #value_throttled

        elements[ksrc]['dropdown'].on_event('menu_item_click', partial(dropdown_callback, source=vsrc, particles=ksrc))


        ####### define layout ################################################################
        blank1 = bmd.Div(width=1000, height=100, text='')
        blank2 = bmd.Div(width=70, height=100, text='')

        if mode == 'ev':
            first_row = [elements[ksrc]['dropdown'], elements[ksrc]['textinput'],
                         blank2, slider]
        else:
            first_row = [slider]
            
        lay = layout([first_row,
                      #[p_cells, p_uv, p_xy],
                      #[p_xVSz, p_yVSz, p_yVSx],
                      [p_cells],
                      [blank1],
                      ])
        tab = bmd.TabPanel(child=lay, title=ksrc)
        tabs.append(tab)
        # end for loop

    doc.add_root(bmd.Tabs(tabs=tabs))
    
parser = argparse.ArgumentParser(description='')
FLAGS = parser.parse_args()
display()
