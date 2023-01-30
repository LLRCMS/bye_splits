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

mode = 'ev'
reprocess = False

data_part_opt = dict(tag='mytag', reprocess=reprocess, debug=True, logger=log)
data_particle = {
    'photons': EventDataParticle(particles='photons', **data_part_opt),
    'electrons': EventDataParticle(particles='electrons', **data_part_opt)}
geom_data = GeometryData(inname='test_triggergeom.root',
                         reprocess=False, logger=log)

def common_props(p):
    p.output_backend = 'svg'
    p.toolbar.logo = None
    # p.grid.visible = False
    # p.outline_line_color = None
    # p.xaxis.visible = False
    # p.yaxis.visible = False

def get_data(particles, event=None):
    region = None
    if mode == 'ev':
        assert region is None
    ds_geom = geom_data.provide(region=region)

    vev = common.dot_dict(cfg_data['varEvents'])
    if mode=='ev':
        if event is None:
            ds_ev, rand_ev = data_particle[particles].provide_random_event()
        else:
            ds_ev = data_particle[particles].provide_event(event)
            rand_ev = -1
        ds_ev['tc'] = ds_ev['tc'].rename(columns={vev.tc['wu']: 'waferu',
                                                  vev.tc['wv']: 'waferv',
                                                  vev.tc['cu']: 'triggercellu',
                                                  vev.tc['cv']: 'triggercellv',
                                                  vev.tc['l']: 'layer', })
        ds_ev['tc'] = ds_ev['tc'].rename(columns={vev.tc['wu']: 'waferu',
                                                  vev.tc['wv']: 'waferv',
                                                  vev.tc['l']: 'layer'})
        #ds_ev['tc'] = ds_ev['tc'].groupby(['layer', 'waferu', 'waferv']).agg(list)
        # ds_ev['tc'] = ds_ev['tc'].groupby(['layer', 'waferu', 'waferv']).agg(list)
        # ds_ev = pd.merge(left=ds_ev['tc'], right=ds_ev['tc'], how='inner',
        #                  on=['layer', 'waferu', 'waferv'])

        ds_ev = pd.merge(left=ds_ev['tc'], right=ds_geom, how='inner',
                         on=['layer', 'waferu', 'waferv', 'triggercellu', 'triggercellv'])
        return {'ev': ds_ev, 'geom': ds_geom, 'rand_ev': rand_ev}

    else:
        return {'geom': ds_geom}

if mode=='ev':
    def_evs = cfg_data['defaultEvents']
    def_ev_text = {}
    for k in def_evs:
        drop_text = [(str(q),str(q)) for q in def_evs[k]]
        def_ev_text[k] = drop_text

    ev_txt = '<p style="width: 60px; padding: 10px; border: 1px solid black;">{}</p>'

widg, cds_data = ({} for _ in range(2))
for k in (data_particle.keys() if mode=='ev' else ('Geometry',)):
    evs = def_evs[k][0] if mode == 'ev' else ''
    cds_data[k] = get_data(k, evs)[mode]

    widg[k] = {'source': bmd.ColumnDataSource(data=cds_data[k])}
    wopt = dict(height=40,)
    if mode=='ev':
        widg[k].update({'textinput': bmd.TextInput(placeholder=str(def_evs[k][0]),
                                                   sizing_mode='stretch_width', **wopt),
                        'dropdown': bmd.Dropdown(label='Default Events',
                                                 button_type='primary',
                                                 menu=def_ev_text[k], **wopt),
                        'button': bmd.Button(label='Random Event', button_type='danger',
                                             **wopt),
                        'text': bmd.Div(text=ev_txt.format(str(def_evs[k][0])), **wopt)})

def range_callback(fig, source, xvar, yvar, shift):
    """18 (centimeters) makes sures entire modules are always within the area shown"""
    fig.x_range.start = min(source.data[xvar]-shift)
    fig.x_range.end = max(source.data[xvar]+shift)
    fig.y_range.start = min(source.data[yvar]-shift)
    fig.y_range.end = max(source.data[yvar]+shift)

def text_callback(attr, old, new, source, figs, particles, border):
    print('text callback ', particles, new)
    if not new.isdecimal():
        print('Wrong format!')
    else:
        source.data = get_data(particles, int(new))[mode]
    for fig in figs:
        range_callback(fig, source, 'tc_x', 'tc_y', border)

def dropdown_callback(event, source, figs, particles, border):
    print('dropdown callback', particles, int(event.__dict__['item']))
    source.data = get_data(particles, int(event.__dict__['item']))[mode]
    for fig in figs:
        range_callback(fig, source, 'tc_x', 'tc_y', border)

def button_callback(event, pretext, source, figs, particles, border):
    print('button callback', particles)
    gd = get_data(particles)
    source.data = gd[mode]
    rand_ev = gd['rand_ev']
    pretext.text = ev_txt.format(str(rand_ev))
    for fig in figs:
        range_callback(fig, source, 'tc_x', 'tc_y', border)

def display():
    doc = curdoc()
    doc.title = 'HGCal Visualization'
    
    width, height   = 600, 500
    tabs = []

    vev = common.dot_dict(cfg_data['varEvents'])
    
    for ksrc,vsrc in [(k,v['source']) for k,v in widg.items()]:
        if mode == 'ev':
            mapper_diams = bmd.LinearColorMapper(palette=mypalette,
                                                 low=vsrc.data[vev.tc['en']].min(),
                                                 high=vsrc.data[vev.tc['en']].max())
            mapper_mods = bmd.LinearColorMapper(palette=mypalette,
                                                low=vsrc.data[vev.tc['en']].min(),
                                                high=vsrc.data[vev.tc['en']].max())

        sld_opt = dict(bar_color='red', width=width, background='white')
        sld_layers = bmd.Slider(start=vsrc.data['layer'].min(), end=vsrc.data['layer'].max(),
                                value=9 if vsrc.data['layer'].max() > 9 else vsrc.data['layer'].min(),
                                step=2, title='Layer', **sld_opt)
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
            sld_en = bmd.Slider(start=cfg_prod['selection']['mipThreshold'], end=5,
                                value=cfg_prod['selection']['mipThreshold'], step=0.1,
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
            all_filters = filt_layers & filt_en
        else:
            all_filters = filt_layers

        view_cells = bmd.CDSView(filter=all_filters)
        # modules are duplicated for cells lying in the same wafer
        # we want to avoid drawing the same module multiple times
        view_modules = (~cds_data[ksrc].duplicated(subset=['layer', 'waferu', 'waferv'])).tolist()
        #view_modules = bmd.CDSView(filter=filt_layers & bmd.BooleanFilter(view_modules))
        view_modules = bmd.CDSView(filter=all_filters)

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

        if mode == 'ev':
            zip_obj = (vsrc.data['diamond_x'],vsrc.data['diamond_y'],vsrc.data['good_tc_mipPt'])
        else:
            zip_obj = (vsrc.data['diamond_x'],vsrc.data['diamond_y'])
        for elem in zip(*zip_obj):
            if mode == 'ev' and elem[2] < cfg_prod['selection']['mipThreshold']: #cut replicates the default `view_en`
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
                       toolbar_location='right', output_backend='webgl')
        border = 18
        p_diams = figure(
            x_range=bmd.Range1d(cur_xmin-border, cur_xmax+border),
            y_range=bmd.Range1d(cur_ymin-border, cur_ymax+border),
            **fig_opt)
        p_mods = figure(
            x_range=p_diams.x_range, y_range=p_diams.y_range,
            **fig_opt)

        if mode == 'ev':
            hover_key_cells = 'Energy (cu,cv / wu,wv)'
            hover_val_cells = '@' + vev.tc['en'] + ' (@triggercellu,@triggercellv / @waferu,@waferv)'
            hover_key_mods = 'Energy (wu,wv)'
            hover_val_mods = '@' + vev.tc['en'] + ' (@'+vev.tc['wu']+',@'+vev.tc['wv']+')'
        else:
            hover_key_cells = 'cu,cv / wu,wv'
            hover_val_cells = '@triggercellu,@triggercellv / @waferu,@waferv'
            hover_key_mods = 'wu,wv'
            #hover_val_mods = '@waferu{custom},@waferv{custom}'
            hover_val_mods = '@'+vev.tc['wu']+',@'+vev.tc['wu']

        hover_code = """
        var wcoord = special_vars.{};
        return wcoord[0];
        """
        
        tool_list = (bmd.BoxZoomTool(match_aspect=True),)
        p_diams.add_tools(bmd.HoverTool(tooltips=[(hover_key_cells, hover_val_cells),],), *tool_list)
        p_mods.add_tools(bmd.HoverTool(tooltips=[(hover_key_mods, hover_val_mods),],), *tool_list)
                                       # formatters={'@waferu': bmd.CustomJSHover(code=hover_code.format('waferu')),
                                       #             '@waferv': bmd.CustomJSHover(code=hover_code.format('waferv'))}

        common_props(p_diams)
        common_props(p_mods)

        polyg_opt = dict(line_color='black', line_width=2)
        p_diams_opt = dict(xs='diamond_x', ys='diamond_y', source=vsrc, view=view_cells, **polyg_opt)
        p_mods_opt = dict(xs='hex_x', ys='hex_y', source=vsrc, view=view_modules, **polyg_opt)
        hover_opt = dict(hover_fill_color='black', hover_line_color='black', hover_line_width=4, hover_alpha=0.2)

        if mode == 'ev':
            p_diams.multi_polygons(fill_color={'field': vev.tc['en'], 'transform': mapper_diams},
                                   **hover_opt, **p_diams_opt)
            p_mods.multi_polygons(fill_color={'field': vev.tc['en'], 'transform': mapper_mods},
                                   **hover_opt, **p_mods_opt)

        else:
            p_diams.multi_polygons(color='green', **hover_opt, **p_diams_opt)
            circ_opt = dict(source=vsrc, view=view_cells, size=5)
            p_diams.circle(x='tc_x', y='tc_y', color='blue', legend_label='u,v conversion', **circ_opt)
            p_diams.circle(x='x', y='y', color='orange', legend_label='tc original', **circ_opt)

            p_mods.multi_polygons(color='green', **hover_opt, **p_mods_opt)
                        
        if mode == 'ev':
            cbar_opt = dict(ticker=bmd.BasicTicker(desired_num_ticks=int(len(mypalette)/4)),
                            formatter=bmd.PrintfTickFormatter(format="%d"))
            cbar_diams = bmd.ColorBar(color_mapper=mapper_diams, title='TC energy [mipPt]', **cbar_opt)
            cbar_mods = bmd.ColorBar(color_mapper=mapper_mods, title='Module Sums [mipPt]', **cbar_opt)

            p_diams.add_layout(cbar_diams, 'right')
            p_mods.add_layout(cbar_mods, 'right')

            widg_opt = dict(source=vsrc, figs=(p_diams, p_mods), particles=ksrc, border=border)
            widg[ksrc]['textinput'].on_change('value', partial(text_callback, **widg_opt))           
            widg[ksrc]['dropdown'].on_event('menu_item_click', partial(dropdown_callback, **widg_opt))
            widg[ksrc]['button'].on_event('button_click', partial(button_callback, pretext=widg[ksrc]['text'], **widg_opt))
            col_names = {'waferu': 'Wafer U', 'waferv': 'Wafer V',
                         'triggercellu': 'Trigger Cell U', 'triggercellv': 'Trigger Cell V',
                         'good_tc_mipPt': 'Energy [mipPt]'}

            template_ints="""<b><div><%= (value).toFixed(0) %></div></b>"""
            template_floats="""<b><div><%= (value).toFixed(3) %></div></b>"""
            formatter_ints = bmd.HTMLTemplateFormatter(template=template_ints)
            formatter_floats = bmd.HTMLTemplateFormatter(template=template_floats)
            cols_diams = [bmd.TableColumn(field=x, title=col_names[x],
                                          formatter=formatter_floats if x=='good_tc_mipPt' else formatter_ints)
                          for x in ['good_tc_mipPt', 'waferu', 'waferv', 'triggercellu', 'triggercellv']]
            cols_mods = [bmd.TableColumn(field=x, title=col_names[x],
                                         formatter=formatter_floats if x=='good_tc_mipPt' else formatter_ints)
                         for x in ['good_tc_mipPt', 'waferu', 'waferv']]
            table_opt = dict(width=width, height=int(0.7*height), source=vsrc)
            table_diams = bmd.DataTable(columns=cols_diams, view=view_cells, **table_opt)
            table_mods = bmd.DataTable(columns=cols_mods, view=view_cells, **table_opt)

            ####### (x,y) plots ################################################################
            coord_opt = dict(width=int(0.66*width), height=int(0.65*width),
                             tools='save,reset', toolbar_location='right',
                             output_backend='webgl')
            coord_widg = (bmd.BoxZoomTool(match_aspect=True),)
            p_xy = figure(**coord_opt, x_axis_label='X [cm]', y_axis_label='Y [cm]')
            p_xy.add_tools(*coord_widg)
            common_props(p_xy)
            p_xy.scatter(x='x', y='y', source=vsrc, view=view_cells,
                         fill_color='blue', line_color='black',)

            # ####### x vs. z plots ################################################################
            p_xz = figure(**coord_opt, x_axis_label='Z [cm]', y_axis_label='X [cm]',)
            p_xz.add_tools(*coord_widg)
            common_props(p_xz)
            p_xz.scatter(x='z', y='x', source=vsrc, view=view_cells,
                         fill_color='blue', line_color='black',)
        
            # ####### y vs. z plots ################################################################
            p_yz = figure(**coord_opt, x_axis_label='Z [cm]', y_axis_label='Y [cm]')
            p_yz.add_tools(*coord_widg)
            common_props(p_yz)
            p_yz.scatter(x='z', y='y', source=vsrc, view=view_cells,
                         fill_color='blue', line_color='black',)
                
        ####### define layout ################################################################
        blank1 = bmd.Div(width=30, height=20, text='')
        gh = '<a href="https://github.com/bfonta/"><img src="display/static/images/github-mark.png" width="20"></a>'
        li = '<a href="https://www.linkedin.com/in/bruno-alves-/"><img src="display/static/images/In-Blue-21@2x.png" width="20"></a>'
        mail = '<a href="mailto:bruno.alves@cern.ch"><img src="display/static/images/MailLogo.png" width="20"></a>'
        signature = bmd.Div(width=500, height=40, text='\u00A9'+' 2023 Bruno Alves' + (5*' ').join(('', gh, li, mail)))

        if mode == 'ev':
            first_row = [widg[ksrc]['dropdown'], widg[ksrc]['textinput'], blank1,
                         widg[ksrc]['button'], widg[ksrc]['text']]
            lay = layout([first_row,
                          sld_layers,
                          sld_en,
                          #[p_xVSz, p_yVSz, p_yVSx],
                          [p_diams, p_mods],
                          [table_diams, table_mods],
                          [blank1],
                          [p_xy, p_xz, p_yz],
                          [blank1],
                          [signature],
                          ])
        else:
            first_row = [sld_layers]            
            lay = layout([first_row,
                          #[p_diams, p_uv, p_xy],
                          #[p_xVSz, p_yVSz, p_yVSx],
                          [p_diams, p_mods],
                          [blank1],
                          [signature],
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
