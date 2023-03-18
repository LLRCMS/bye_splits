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

def calc_universal_coordinates(df, varu='univ_u', varv='univ_v'):
    nside = 4
    df[varu] = df.tc_cu - nside*df.tc_wu + 2*nside*df.tc_wv
    df[varv] = df.tc_cv - 2*nside*df.tc_wu + nside*df.tc_wv
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
                 'tc_mipPt': 'Energy [mipPt]'}

    template_ints="""<b><div><%= (value).toFixed(0) %></div></b>"""
    template_floats="""<b><div><%= (value).toFixed(3) %></div></b>"""
    fi = bmd.HTMLTemplateFormatter(template=template_ints)
    ff = bmd.HTMLTemplateFormatter(template=template_floats)
    cols = [bmd.TableColumn(field=x, title=col_names[x], formatter=fi)
            for x in ['tc_wu', 'tc_wv', 'tc_cu', 'tc_cv']]
    cols.extend([bmd.TableColumn(field='tc_mipPt', title=col_names['tc_mipPt'],
                                 formatter=ff)])
    table_opt = dict(width=width, height=int(0.7*height), source=src)
    table = bmd.DataTable(columns=cols, **table_opt)
    return table
            
def roi_event_loop(pars, **kw):
    in_tcs = common.fill_path(kw['SeedIn']+'_'+kw['FesAlgo'], **pars)
    uv_vars = ['univ_u', 'univ_v', 'tc_cu', 'tc_cv', 'tc_wu', 'tc_wv']
    tabs_one, tabs_many = ([] for _ in range(2))
    
    nn = 0
    ntabs = 10
    if FLAGS.fullroi:
        # randomly center on wafer (3,5)
        allowed = ('35', '36', '25' ,'24', '34', '45', '45', '46')
        arrcv = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,
                          4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,
                          0,1,2,3,1,2,3,4,2,3,4,5,3,4,5,6])
        arrcu = np.array([1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7,
                          4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7,
                          0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3])
        assert arrcv.shape==arrcu.shape

        toy = pd.DataFrame()
        for wu in range(2,5):
            for wv in range(4,7):
                if str(wu)+str(wv) in allowed:
                    arrwu = np.repeat(wu,arrcu.shape[0])
                    arrwv = np.repeat(wv,arrcv.shape[0])
                    toytmp = pd.DataFrame({'tc_wu': arrwu,
                                           'tc_wv': arrwv,
                                           'tc_cu': arrcu,
                                           'tc_cv': arrcv})
                    toy = pd.concat((toy,toytmp), axis=0)
                    

        toy = toy.reset_index()[toy.columns]
        toy = calc_universal_coordinates(toy, varu=uv_vars[0], varv=uv_vars[1])
        lay = create_roi_tab(toy)
        panel = bmd.TabPanel(child=lay, title='Full ROI')
        plot_all_rois([panel], 'uv_to_square_fullroi.html')
        
    else:
        store_roi = pd.HDFStore(in_tcs, mode='r')
        unev = store_roi.keys()
        for key in unev:
            roi_ev = store_roi[key]
            roi_ev = calc_universal_coordinates(roi_ev, varu=uv_vars[0], varv=uv_vars[1])
     
            if len(roi_ev['tc_wu'].unique())==1 and len(roi_ev['tc_wv'].unique())==1:
                neighbours = False
            else:
                neighbours = True
                nn += 1
            roi_ev = roi_ev.groupby(by=uv_vars).sum()[['tc_mipPt']].reset_index()
     
            lay = create_roi_tab(roi_ev)
            tab_title = re.findall('\d+', key)[0]
            panel = bmd.TabPanel(child=lay, title=tab_title)
            if neighbours==True:
                if len(tabs_many) < ntabs:
                    tabs_many.append(panel)
            else:
                if len(tabs_one) < ntabs:
                    tabs_one.append(panel)
     
            if len(tabs_one) > ntabs and len(tabs_many) > ntabs:
                break
            
        store_roi.close()
        print('There are {}/{} ROIs with neighbours.'.format(nn,len(unev)))
        
        plot_all_rois(tabs_one, 'uv_to_square_one.html')
        plot_all_rois(tabs_many, 'uv_to_square_many.html')

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

def create_roi_tab(df):
    width, height = 800, 800
    title = r''
    basic_tools = 'pan,save,reset,undo,box_select'

    df = add_bokeh_coord_convention(df, 'univ_u', 'univ_v')
    source = bmd.ColumnDataSource(df)
    hexq = source.data['q_bokeh']
    hexr = source.data['r_bokeh']
    hexcu = source.data['univ_u']
    hexcv = source.data['univ_v']
    hexsize = .5

    xmin, xmax, ymin, ymax = source_maxmin(source, 'univ_u', 'univ_v')
    p_uv = figure(width=width, height=height,
                  title='ROI TCs u/v coordinates',
                  tools=basic_tools, toolbar_location='right',
                  output_backend='webgl',
                  x_range=bmd.Range1d(xmin, xmax),
                  y_range=bmd.Range1d(ymin, ymax))
    common_props(p_uv)

    hover_opt = dict(hover_fill_color='black', hover_line_color='black',
                     hover_line_width=4, hover_alpha=0.2)

    uv_r = p_uv.hex_tile(q='q_bokeh', r='r_bokeh', orientation='flattop',
                         source=source,
                         size=hexsize, fill_color='green', line_color='black',
                         line_width=3, alpha=1., **hover_opt)
    hvr_tt = [('cu,cv/wu,wv', '@tc_cu,@tc_cv/@tc_wu,@tc_wv')]
    p_uv.add_tools(bmd.WheelZoomTool(),
                   bmd.HoverTool(tooltips=hvr_tt, renderers=[uv_r]))

    hexx, hexy = axial_to_cartesian(q=hexq, r=hexr, size=hexsize,
                                    orientation='flattop')
    p_uv.text(hexx, hexy,
              text=['{},{}'.format(q,r) for (q, r) in zip(hexcu, hexcv)],
              text_baseline='middle', text_align='center',
              text_font_size='8pt' if FLAGS.fullroi else '10pt')

    p_xy = figure(width=width, height=height,
                  title='ROI TCs transfer to a square grid',
                  tools=basic_tools, toolbar_location='right',
                  output_backend='webgl',
                  x_range=bmd.Range1d(xmin, xmax),
                  y_range=bmd.Range1d(ymin, ymax))
    common_props(p_xy)
    xy_r = p_xy.rect(x='univ_u', y='univ_v', source=source,
                     width=1., height=1.,
                     width_units='data', height_units='data',
                     fill_color='red', line_color='black', line_width=3,
                     **hover_opt)
    p_xy.text(hexcu, hexcv,
              text=['{},{}'.format(q,r) for (q, r) in zip(hexcu, hexcv)],
              text_baseline='middle', text_align='center',
              text_font_size='9pt' if FLAGS.fullroi else '11pt')
    p_xy.add_tools(bmd.WheelZoomTool(),
                   bmd.HoverTool(tooltips=hvr_tt, renderers=[xy_r]))
    
    table = create_bokeh_datatable(source, width, height)
    
    return layout([[p_uv, p_xy], [table]])

def plot_all_rois(tabs, out):
    adir = '/eos/user/b/bfontana/www/L1/SeedROIStudies/'
    output_file(os.path.join(adir, out))
    save(bmd.Tabs(tabs=tabs))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Study ROI UV/grid distributions')
    parser.add_argument('--fullroi', action='store_true',
                        help='display an toy full region of interest')
    parsing.add_parameters(parser) 
    FLAGS = parser.parse_args()
 
    seed_d = params.read_task_params('seed_roi')
    roi_event_loop(vars(FLAGS), **seed_d)
