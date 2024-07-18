# coding: utf-8

_all_ = []

import os
import sys
parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

import re
import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import layout
from bokeh import models as bmd
from bokeh.plotting import figure
from bokeh.palettes import viridis as _palette
from bokeh.util.hex import axial_to_cartesian

import utils
from utils import params, common, parsing

import tasks
from tasks.seed_cs import dist

def add_centrals_info(cs_ev, centr):
    """
    A negative number indicates a wafer that is not the cente rof any CS in the event.
    A positive number indicates a wafer that is the center of a CS in the event.
    Zero is not assigned.
    """
    cs_ev['central_cs'] = 0
    for ic,c in enumerate(centr):
        iscentral = (cs_ev.tc_wu==c[0]) & (cs_ev.tc_wv==c[1])
        cs_ev.loc[iscentral, 'central_cs'] = ic+1
        
    non_centrals = cs_ev[cs_ev.central_cs == 0]
    non_centrals = non_centrals[['tc_wu', 'tc_wv']].drop_duplicates().to_numpy()
    for ip,pair in enumerate(non_centrals):
        sel = (cs_ev.tc_wu==pair[0]) & (cs_ev.tc_wv==pair[1])
        cs_ev.loc[sel, 'central_cs'] = -ip-1
    return cs_ev

def coord_transf(cu, wu, cv, wv):
    nside = 4
    return cu - nside*wu + 2*nside*wv, cv - 2*nside*wu + nside*wv
    
def calc_universal_coordinates(df, varu='univ_u', varv='univ_v'):
    nside = 4
    df[varu], df[varv] = coord_transf(df.tc_cu, df.tc_wu, df.tc_cv, df.tc_wv)
    df[varu] = df[varu] - df[varu].min()
    df[varv] = df[varv] - df[varv].min()
    return df
    
def common_props(p):
    p.output_backend = 'svg'
    p.toolbar.logo = None
    p.grid.visible = False
    p.outline_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

def create_bokeh_datatable(src, width, height):
    col_names = {'tc_wu': 'Wafer U',
                 'tc_wv': 'Wafer V',
                 'tc_cu': 'Trigger Cell U',
                 'tc_cv': 'Trigger Cell V',
                 'tc_mipPt': 'Energy [mipPt]',
                 'tc_energy': 'Energy [GeV]'}

    template_ints="""<b><div><%= (value).toFixed(0) %></div></b>"""
    template_floats="""<b><div><%= (value).toFixed(3) %></div></b>"""
    fi = bmd.HTMLTemplateFormatter(template=template_ints)
    ff = bmd.HTMLTemplateFormatter(template=template_floats)
    cols = [bmd.TableColumn(field=x, title=col_names[x], formatter=fi)
            for x in ['tc_wu', 'tc_wv', 'tc_cu', 'tc_cv']]
    cols.extend([bmd.TableColumn(field='tc_mipPt', title=col_names['tc_mipPt'],
                                 formatter=ff)])
    cols.extend([bmd.TableColumn(field='tc_energy', title=col_names['tc_energy'],
                                 formatter=ff)])
    table_opt = dict(width=width, height=int(0.7*height), source=src)
    table = bmd.DataTable(columns=cols, **table_opt)
    return table

def dist(u1, v1, u2, v2):
    """distance in an hexagonal grid"""
    s1 = u1 - v1
    s2 = u2 - v2
    return (abs(u1-u2) + abs(v1-v2) + abs(s1-s2)) / 2
    
def cs_event_loop(pars, **kw):
    uv_vars = ['univ_u', 'univ_v', 'tc_cu', 'tc_cv', 'tc_wu', 'tc_wv']
    tabs = []
    
    ntabs = 18
    if FLAGS.fullcs:
        # randomly center on wafer (3,5)
        center = '35'
        allowed = (center, '36', '25' ,'24', '34', '45', '45', '46')

        # cell indices of TCs in the wafer
        arrcu = np.array([1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7,
                          4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7,
                          0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
        arrcv = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,
                          4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
                          0,1,2,3,1,2,3,4,2,3,4,5,3,4,5,6])

        # cell indices of TCs in the border of the wafer
        uborder = np.array([1,2,3,4,5,6,7,7,7,7,7,6,5,4,3,2,1,0,0,0,0])
        vborder = np.array([0,0,0,0,1,2,3,4,5,6,7,7,7,7,6,5,4,3,2,1,0])
        # universal coordinates of TCs in the center wafer's border
        uborder, vborder = coord_transf(uborder, int(center[0]), vborder, int(center[1]))

        assert arrcv.shape==arrcu.shape

        toy = pd.DataFrame()
        for wu in range(2,5):
            for wv in range(4,7):
                wafer = str(wu)+str(wv)
                if wafer not in allowed:
                    continue

                if wafer == center:
                    colors = np.full(arrcv.shape, 'saddlebrown')
                else:
                    colors = np.full(arrcv.shape, 'bisque')

                    # universal coordinates of TCs in neighbouring wafer
                    uneigh, vneigh = coord_transf(arrcu, wu, arrcv, wv)

                    # change neighbours' color
                    for ub, vb in zip(uborder, vborder):
                        dists = dist(uneigh, vneigh, ub, vb)
                        colors[dists <= 1] = 'cyan'

                toytmp = pd.DataFrame({'tc_wu': np.repeat(wu,arrcu.shape[0]),
                                       'tc_wv': np.repeat(wv,arrcv.shape[0]),
                                       'tc_cu': arrcu,
                                       'tc_cv': arrcv,
                                       'colors': colors,
                                       })
                toy = pd.concat((toy,toytmp), axis=0)

        toy = toy.reset_index()[toy.columns]
        toy = calc_universal_coordinates(toy, varu=uv_vars[0], varv=uv_vars[1])
        lay = create_cs_tab(toy)
        panel = bmd.TabPanel(child=lay, title='Full CS')
        plot_all_css([panel], 'uv_to_square_fullcs.html')
        
    else:
        in_tcs = common.fill_path(kw['SeedIn'], **pars)
        store_cs = pd.HDFStore(in_tcs, mode='r')
        unev = [x for x in store_cs.keys() if 'central' not in x]
        inspect_events = ('165405',)
        for key in unev:
            cs_ev = store_cs[key]
            cs_uvcentrals = store_cs[key + 'central'].to_numpy()
            
            cs_ev = calc_universal_coordinates(cs_ev, varu=uv_vars[0], varv=uv_vars[1])
            cs_ev = cs_ev.groupby(by=uv_vars).sum()[['tc_mipPt', 'tc_energy']].reset_index()
            cs_ev = add_centrals_info(cs_ev, cs_uvcentrals)

            lay = create_cs_tab(cs_ev)
            tab_title = re.findall('.*_([0-9]{1,7})_.*', key)[0]
            panel = bmd.TabPanel(child=lay, title=tab_title)
            if len(tabs) < ntabs or any([x in key for x in inspect_events]):
                tabs.append(panel)

            if '165405' in key:
                inspected = True

            if len(tabs) > ntabs and inspected:
                break
            
        store_cs.close()
        
        plot_all_css(tabs, 'uv_to_square.html')

def source_maxmin(src, varu, varv):
    # find dataset minima and maxima
    xmax, ymax = (-1e9 for _ in range(2))
    xmin, ymin = (+1e9 for _ in range(2))
    zobj = src.data[varu], src.data[varv]
    
    for elem in zip(*zobj):
        if elem[0] > xmax: xmax = elem[0]
        if elem[0] < xmin: xmin = elem[0]
        if elem[1] > ymax: ymax = elem[1]
        if elem[1] < ymin: ymin = elem[1]

    # force matching ratio to avoid distortions
    distx, disty = xmax-xmin, ymax-ymin
    if distx > disty:
        ymax += abs(distx-disty)/2
        ymin = ymax - distx
    else:
        xmin -= abs(distx-disty)/2
        xmax = xmin + disty
    xmax += (xmax-xmin)*0.15
    xmin -= (xmax-xmin)*0.15
    ymax += (ymax-ymin)*0.15
    ymin -= (ymax-ymin)*0.15
    return xmin, xmax, ymin, ymax

def add_bokeh_coord_convention(df, varu, varv):
    df['r_bokeh'] = -df[varu]
    df['q_bokeh'] = df[varv]
    return df

def create_cs_tab(df):
    width, height = 1200, 1200
    title = r''
    basic_tools = 'pan,save,reset,undo,box_select'

    df = add_bokeh_coord_convention(df, 'univ_u', 'univ_v')
    if not FLAGS.fullcs:
        mypalette = _palette(50)
        mapper = bmd.LinearColorMapper(palette=mypalette,
                                       low=df.tc_mipPt.min(), high=df.tc_mipPt.max())
        cbar_opt = dict(ticker=bmd.BasicTicker(desired_num_ticks=int(len(mypalette)/4)),
                        formatter=bmd.PrintfTickFormatter(format="%d"))
        cbar = bmd.ColorBar(color_mapper=mapper, title='TC energy [mipPt]', **cbar_opt)

    source = bmd.ColumnDataSource(df)
    hexq = source.data['q_bokeh']
    hexr = source.data['r_bokeh']
    hexcu = source.data['univ_u']
    hexcv = source.data['univ_v']
    hexsize = .5

    xmin, xmax, ymin, ymax = source_maxmin(source, 'univ_u', 'univ_v')
    p_uv = figure(width=width, height=height,
                  title='CS TCs u/v coordinates',
                  tools=basic_tools, toolbar_location='right',
                  output_backend='webgl',
                  x_range=bmd.Range1d(xmin, xmax),
                  y_range=bmd.Range1d(ymin, ymax))
    common_props(p_uv)

    hover_opt = dict(hover_fill_color='black', hover_line_color='black',
                     hover_line_width=4, hover_alpha=0.2)

    uv_r = p_uv.hex_tile(q='q_bokeh', r='r_bokeh', orientation='flattop',
                         source=source,
                         size=hexsize,
                         fill_color=('colors' if FLAGS.fullcs
                                     else {'field': 'tc_mipPt', 'transform': mapper}),
                         line_color='black',
                         line_width=3, alpha=1., **hover_opt)
    if not FLAGS.fullcs:
        p_uv.add_layout(cbar, 'right')

    if FLAGS.fullcs:
        hvr_tt = [('cu,cv/wu,wv', '@tc_cu,@tc_cv/@tc_wu,@tc_wv')]
    else:
        hvr_tt = [('mipPt [mipPt] / energy [GeV] / CS id', '@tc_mipPt / @tc_energy / @central_cs')]
    p_uv.add_tools(bmd.WheelZoomTool(),
                   bmd.HoverTool(tooltips=hvr_tt, renderers=[uv_r]))

    hexx, hexy = axial_to_cartesian(q=hexq, r=hexr, size=hexsize,
                                    orientation='flattop')
    p_uv.text(hexx, hexy,
              text=['{},{}'.format(q,r) for (q, r) in zip(hexcu, hexcv)],
              text_baseline='middle', text_align='center',
              text_font_size='8pt' if FLAGS.fullcs else '10pt')

    p_xy = figure(width=width, height=height,
                  title='CS TCs transfer to a square grid',
                  tools=basic_tools, toolbar_location='right',
                  output_backend='webgl',
                  x_range=bmd.Range1d(xmin, xmax),
                  y_range=bmd.Range1d(ymin, ymax))
    common_props(p_xy)
    xy_r = p_xy.rect(x='univ_u', y='univ_v', source=source,
                     width=1., height=1.,
                     width_units='data', height_units='data',
                     fill_color=('colors' if FLAGS.fullcs
                                 else {'field': 'tc_mipPt', 'transform': mapper}),
                     line_color='black', line_width=3,
                     **hover_opt)
    if not FLAGS.fullcs:
        p_xy.add_layout(cbar, 'right')
    p_xy.text(hexcu, hexcv,
              text=['{},{}'.format(q,r) for (q, r) in zip(hexcu, hexcv)],
              text_baseline='middle', text_align='center',
              text_font_size='9pt' if FLAGS.fullcs else '11pt')
    p_xy.add_tools(bmd.WheelZoomTool(),
                   bmd.HoverTool(tooltips=hvr_tt, renderers=[xy_r]))
    
    table = create_bokeh_datatable(source, width, height)
    
    return layout([[p_uv, p_xy], [table]])

def plot_all_css(tabs, out):
    adir = '/eos/home-b/bfontana/www/L1/SeedCSStudies/'
    output_file(os.path.join(adir, out))
    save(bmd.Tabs(tabs=tabs))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Study CS UV/grid distributions')
    parser.add_argument('--fullcs', action='store_true',
                        help='display an toy full region of interest')
    parsing.add_parameters(parser) 
    FLAGS = parser.parse_args()
 
    seed_d = params.read_task_params('seed_cs')
    cs_event_loop(vars(FLAGS), **seed_d)
