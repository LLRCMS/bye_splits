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

with open(params.viz_kw['CfgEventPath'], 'r') as afile:
    config = yaml.safe_load(afile)

data_part_opt = dict(tag='v2', reprocess=False, debug=True, logger=log)
data_particle = {
    'photons': EventDataParticle(particles='photons', **data_part_opt),
    'electrons': EventDataParticle(particles='electrons', **data_part_opt)}
geom_data = GeometryData(inname='test_triggergeom_v2.root', reprocess=False, logger=log)
data_vars = {'ev': config['varEvents'],
             'geom': config['varGeometry']}
mode = 'ev'

def common_props(p):
    p.output_backend = 'svg'
    p.toolbar.logo = None
    p.grid.visible = False
    p.outline_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

def rotate(angle, x, y, cx, cy):
    """Counter-clockwise rotation of 'angle' [radians]"""
    assert angle >= 0 and angle < 2 * np.pi
    ret_x = np.cos(angle)*(x-cx) - np.sin(angle)*(y-cy) + cx
    ret_y = np.sin(angle)*(x-cx) + np.cos(angle)*(y-cy) + cy
    return ret_x, ret_y

def convert_cells_to_xy(df):
    c30, s30, t30 = np.sqrt(3)/2, 1/2, 1/np.sqrt(3)
    N = 4
    waferWidth = 1
    R = waferWidth / (3 * N)
    r = R * c30
    cellDistX = waferWidth/8.
    cellDistY = cellDistX * t30

    scu, scv = 'triggercellu', 'triggercellv'
    swu, swv = 'waferu', 'waferv'
    df.loc[:, 'wafer_shift_x'] = (2*df[swu] - df[swv])*waferWidth/2
    df.loc[:, 'wafer_shift_y'] = c30*df[swv]
    
    cells_conversion = (lambda cu,cv: (1.5*(cv-cu)+0.5) * R, lambda cu,cv: (cv+cu-2*N+1) * r) #orientation 6

    univ_wcenterx = (1.5*(3-3) + 0.5)*R + cellDistX
    univ_wcentery = (3 + 3 - 2*N + 1) * r + 3*cellDistY/2
    scale_x, scale_y = waferWidth/2, waferWidth/(2*c30)

    xcorners, ycorners = ([] for _ in range(2))
    xcorners.append(univ_wcenterx - scale_x)
    xcorners.append(univ_wcenterx)
    xcorners.append(univ_wcenterx + scale_x)
    xcorners.append(univ_wcenterx + scale_x)
    xcorners.append(univ_wcenterx)
    xcorners.append(univ_wcenterx - scale_x)
    ysub = np.sqrt(scale_y*scale_y-scale_x*scale_x)
    ycorners.append(univ_wcentery - ysub)
    ycorners.append(univ_wcentery - scale_y)
    ycorners.append(univ_wcentery - ysub)
    ycorners.append(univ_wcentery + ysub)
    ycorners.append(univ_wcentery + scale_y)
    ycorners.append(univ_wcentery + ysub)

    def masks_location(location, ax, ay):
        """Filter TC location in wafer: up-right, up-left and bottom.
        The x/y location depends on the wafer orientation."""
        ux = np.sort(ax.unique())
        uy = np.sort(ay.unique())
        if len(ux)==0 or len(uy)==0:
            return pd.Series(dtype=float)
        
        if len(ux) != 2*N: #full wafers
            m = 'Length unique X values vs expected for full wafers: {} vs {}\n'.format(len(ux), 2*N)
            m += 'Fix.'
            raise AssertionError(m)
        if len(uy) != 4*N-1: #full wafers
            m = 'Length unique Y values vs expected for full wafers: {} vs {}\n'.format(len(uy), 4*N-1)
            m += 'Fix.'
            raise AssertionError(m)

        b = (-1/12, 0.)
        fx, fy = 1/8, (1/8)*t30 #multiplicative factors: cells are evenly spaced
        eps = 0.02 #epsilon, create an interval around the true values
        cx = abs(round((ux[0]-b[0])/fx)) 
        cy = abs(round((uy[N-1]-b[1])/fy))
        # -0.216, -0.144, -0.072, -0.000 /// +0.072, -0.000, -.072, -0.144

        filt_UL = ((ax > b[0]-(cx-0)*fx-eps) & (ax < b[0]-(cx-0)*fx+eps) & (ay > b[1]-(cy-0)*fy-eps) |
                   (ax > b[0]-(cx-1)*fx-eps) & (ax < b[0]-(cx-1)*fx+eps) & (ay > b[1]-(cy-1)*fy-eps) |
                   (ax > b[0]-(cx-2)*fx-eps) & (ax < b[0]-(cx-2)*fx+eps) & (ay > b[1]-(cy-2)*fy-eps) |
                   (ax > b[0]-(cx-3)*fx-eps) & (ax < b[0]-(cx-3)*fx+eps) & (ay > b[1]-(cy-3)*fy-eps))
                                                                                            
        filt_UR = ((ax > b[0]-(cx-4)*fx-eps) & (ax < b[0]-(cx-4)*fx+eps) & (ay > b[1]-(cy-4)*fy-eps) |
                   (ax > b[0]-(cx-5)*fx-eps) & (ax < b[0]-(cx-5)*fx+eps) & (ay > b[1]-(cy-3)*fy-eps) |
                   (ax > b[0]-(cx-6)*fx-eps) & (ax < b[0]-(cx-6)*fx+eps) & (ay > b[1]-(cy-2)*fy-eps) |
                   (ax > b[0]-(cx-7)*fx-eps) & (ax < b[0]-(cx-7)*fx+eps) & (ay > b[1]-(cy-1)*fy-eps))

        if location == 'UL':
            return filt_UL
        elif location == 'UR':
            return filt_UR
        else: #bottom
            return (~filt_UL & ~filt_UR)

    xpoint, x0, x1, x2, x3 = ({} for _ in range(5))
    ypoint, y0, y1, y2, y3 = ({} for _ in range(5))
    xaxis, yaxis = ({} for _ in range(2))

    df.loc[:, 'tc_x_center'] = (1.5*(df[scv]-df[scu])+0.5) * R #orientation 6
    df.loc[:, 'tc_y_center'] = (df[scv]+df[scu]-2*N+1) * r #orientation 6
    df.loc[:, 'tc_x'] = df.wafer_shift_x + df['tc_x_center']
    df.loc[:, 'tc_y'] = df.wafer_shift_y + df['tc_y_center']
    wcenter_x = df.wafer_shift_x + univ_wcenterx # fourth vertex (center) for cu/cv=(3,3)
    wcenter_y = df.wafer_shift_y + univ_wcentery # fourth vertex (center) for cu/cv=(3,3)

    for loc_key in ('UL', 'UR', 'B'):
        masks_loc = masks_location(loc_key, df['tc_x_center'], df['tc_y_center'])
        cx_d, cy_d = df['tc_x'][masks_loc], df['tc_y'][masks_loc]
        wc_x, wc_y = wcenter_x[masks_loc], wcenter_y[masks_loc]

        # x0 refers to the x position the lefmost, down corner all diamonds (TCs)
        # x1, x2, x3 are defined in a counter clockwise fashion
        # same for y0, y1, y2 and y3
        # tc positions refer to the center of the diamonds
        if loc_key == 'UL':
            x0.update({loc_key: cx_d})
        elif loc_key == 'UR':
            x0.update({loc_key: cx_d})
        else:
            x0.update({loc_key: cx_d - cellDistX})
                
        x1.update({loc_key: x0[loc_key][:] + cellDistX})
        if loc_key in ('UL', 'UR'):
            x2.update({loc_key: x1[loc_key]})
            x3.update({loc_key: x0[loc_key]})
        else:
            x2.update({loc_key: x1[loc_key] + cellDistX})
            x3.update({loc_key: x1[loc_key]})

        if loc_key == 'UL':
            y0.update({loc_key: cy_d})
        elif loc_key == 'UR':
            y0.update({loc_key: cy_d})
        else:
            y0.update({loc_key: cy_d + cellDistY})

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

        angle = 0#2*np.pi/3
        x0[loc_key], y0[loc_key] = rotate(angle, x0[loc_key], y0[loc_key], wc_x, wc_y)
        x1[loc_key], y1[loc_key] = rotate(angle, x1[loc_key], y1[loc_key], wc_x, wc_y)
        x2[loc_key], y2[loc_key] = rotate(angle, x2[loc_key], y2[loc_key], wc_x, wc_y)
        x3[loc_key], y3[loc_key] = rotate(angle, x3[loc_key], y3[loc_key], wc_x, wc_y)
            
        keys = ['pos0','pos1','pos2','pos3']
        xaxis.update({
            loc_key: pd.concat([x0[loc_key],x1[loc_key],x2[loc_key],x3[loc_key]],
                               axis=1, keys=keys)})
        yaxis.update(
            {loc_key: pd.concat([y0[loc_key],y1[loc_key],y2[loc_key],y3[loc_key]],
                                axis=1, keys=keys)})

        xaxis[loc_key]['new'] = [[[[round(val, 3) for val in sublst]]]
                                 for sublst in xaxis[loc_key].values.tolist()]
        yaxis[loc_key]['new'] = [[[[round(val, 3) for val in sublst]]]
                                 for sublst in yaxis[loc_key].values.tolist()]
        xaxis[loc_key] = xaxis[loc_key].drop(keys, axis=1)
        yaxis[loc_key] = yaxis[loc_key].drop(keys, axis=1)

    df.loc[:, 'diamond_x'] = pd.concat(xaxis.values())
    df.loc[:, 'diamond_y'] = pd.concat(yaxis.values())

    # define module corners' coordinates
    xcorners_str = ['corner1x','corner2x','corner3x','corner4x','corner5x','corner6x']
    assert len(xcorners_str) == len(xcorners)
    ycorners_str = ['corner1y','corner2y','corner3y','corner4y','corner5y','corner6y']
    assert len(ycorners_str) == len(ycorners)
    for i in range(len(xcorners)):
        df[xcorners_str[i]] = df.wafer_shift_x + xcorners[i]
    for i in range(len(ycorners)):
        df[ycorners_str[i]] = df.wafer_shift_y + ycorners[i]

    df.loc[:, 'hex_x'] = df[xcorners_str].values.tolist()
    df.loc[:, 'hex_x'] = df['hex_x'].map(lambda x: [[x]])
    df.loc[:, 'hex_y'] = df[ycorners_str].values.tolist()
    df.loc[:, 'hex_y'] = df['hex_y'].map(lambda x: [[x]])
    df = df.drop(xcorners_str + ycorners_str + ['tc_x_center', 'tc_y_center'], axis=1)
    return df

def get_data(event, particles):
    ds_geom = geom_data.provide()
    # ds_geom = ds_geom[((ds_geom[data_vars['geom']['wu']]==3) & (ds_geom[data_vars['geom']['wv']]==3)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==3) & (ds_geom[data_vars['geom']['wv']]==4)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==4) & (ds_geom[data_vars['geom']['wv']]==3)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==4) & (ds_geom[data_vars['geom']['wv']]==4))]

    # ds_geom = ds_geom[((ds_geom[data_vars['geom']['wu']]==-6) & (ds_geom[data_vars['geom']['wv']]==3)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==-6) & (ds_geom[data_vars['geom']['wv']]==4)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==-7) & (ds_geom[data_vars['geom']['wv']]==3)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==-8) & (ds_geom[data_vars['geom']['wv']]==2)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==-8) & (ds_geom[data_vars['geom']['wv']]==1)) |
    #                   ((ds_geom[data_vars['geom']['wu']]==-7) & (ds_geom[data_vars['geom']['wv']]==2))
    #                   ]
    #ds_geom = ds_geom[ds_geom.layer<=9]
    ds_geom = ds_geom[ds_geom.waferpart==0]
    ds_geom = convert_cells_to_xy(ds_geom)

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
    with open(params.viz_kw['CfgEventPath'], 'r') as afile:
        cfg = yaml.safe_load(afile)
        def_evs = cfg['defaultEvents']
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
        elements[k].update({'textinput': bmd.TextInput(value='<specify an event>', height=40,
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

    for ksrc,vsrc in [(k,v['source']) for k,v in elements.items()]:

        if mode == 'ev':
            mapper = bmd.LinearColorMapper(palette=mypalette,
                                           low=vsrc.data['good_tc_mipPt'].min(), high=vsrc.data['good_tc_mipPt'].max())

        slider = bmd.Slider(start=vsrc.data['layer'].min(), end=vsrc.data['layer'].max(),
                            value=vsrc.data['layer'].min(), step=2, title='Layer',
                            bar_color='red', width=600, background='white')
        slider_callback = bmd.CustomJS(args=dict(s=vsrc), code="""s.change.emit();""")
        slider.js_on_change('value', slider_callback) #value_throttled

        filter_cells = bmd.CustomJSFilter(args=dict(slider=slider), code="""
           var indices = new Array(source.get_length());
           var sval = slider.value;
    
           const subset = source.data['layer'];
           for (var i=0; i < source.get_length(); i++) {
               indices[i] = subset[i] == sval;
           }
           return indices;
           """)
        view_cells = bmd.CDSView(filter=filter_cells)
        # modules are duplicated for cells lying in the same wafer
        # we want to avoid drawing the same module multiple times
        view_modules = (~cds_data[ksrc].duplicated(subset=['layer', 'waferu', 'waferv'])).tolist()
        #view_modules = bmd.CDSView(filter=filter_cells & bmd.BooleanFilter(view_modules))
        view_modules = bmd.CDSView(filter=filter_cells)

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
        p_diams = figure(x_range=bmd.Range1d(cur_xmin, cur_xmax), y_range=bmd.Range1d(cur_ymin, cur_ymax), **fig_opt)
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
            p_diams.multi_polygons(fill_color={'field': 'good_tc_mipPt', 'transform': mapper},
                                   **hover_opt, **p_diams_opt)
            p_mods.multi_polygons(fill_color={'field': 'good_tc_mipPt', 'transform': mapper}, #CHANGE WHEN MODULE SUMS ARE AVAILABLE
                                   **hover_opt, **p_mods_opt)

        else:
            p_diams.multi_polygons(color='green', **hover_opt, **p_diams_opt)
            p_diams.circle(x='tc_x', y='tc_y', source=vsrc, view=view_cells, size=4, color='red')

            p_mods.multi_polygons(color='green', **hover_opt, **p_mods_opt)
                        
        if mode == 'ev':
            color_bar = bmd.ColorBar(color_mapper=mapper,
                                     ticker=bmd.BasicTicker(desired_num_ticks=int(len(mypalette)/4)),
                                     formatter=bmd.PrintfTickFormatter(format="%d"))
            p_diams.add_layout(color_bar, 'right')
            p_mods.add_layout(color_bar, 'right')

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
        
        ####### text input ###################################################################

        slider_callback = bmd.CustomJS(args=dict(s=vsrc), code="""s.change.emit();""")
        slider.js_on_change('value', slider_callback) #value_throttled

        ####### define layout ################################################################
        blank1 = bmd.Div(width=1000, height=100, text='')
        blank2 = bmd.Div(width=70, height=100, text='')

        if mode == 'ev':
            first_row = [elements[ksrc]['dropdown'], elements[ksrc]['textinput'],
                         blank2, slider]
        else:
            first_row = [slider]
            
        lay = layout([first_row,
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
