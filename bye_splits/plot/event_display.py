# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
import uproot as up

from bokeh.io import output_file, save
from bokeh.plotting import figure
from bokeh.util.hex import axial_to_cartesian
from bokeh.models import (
    Panel,
    Tabs,
    BoxZoomTool,
    Range1d,
    ColumnDataSource,
    HoverTool,
    )
from bokeh.layouts import layout
output_file('tmp.html')

import utils
from utils import params, common, parsing

flatten = lambda col : col.to_numpy().squeeze()

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


tc_path = ( Path(__file__).parent.absolute().parent.parent /
            params.DataFolder / 'test_triggergeom.root' )
tc_file = up.open(tc_path)
tc_tree = tc_file[ os.path.join('hgcaltriggergeomtester', 'TreeTriggerCells') ]
# print(tc_tree.show())

var1, var2 = 'waferu', 'waferv'
sel_vars = 'zside', 'layer', 'subdet'
tc_data = tc_tree.arrays({var1, var2, *sel_vars}, library='pd')
layers = flatten(tc_data[['layer']].drop_duplicates().sort_values('layer'))

tabs_uv, figs_uv = ([] for _ in range(2))
tabs_xy, figs_xy = ([] for _ in range(2))

for il in layers:
    l_data = tc_data[(tc_data.layer==il) & (tc_data.zside==1) & (tc_data.subdet==1)][:]
    l_data = l_data[[var1, var2]].drop_duplicates().sort_values([var1, var2])

    u = flatten(l_data[var1])
    v = flatten(l_data[var2])
    v *= -1 #convert between CMSSW coordinate system and bokeh's
    xcart, ycart = axial_to_cartesian(u, v, 1, 'pointytop')
    source = ColumnDataSource({'u': u, 'v': v, 'xcart': xcart, 'ycart': ycart,
                               'color': ['firebrick']*len(u)})

    # (u,v) plots
    figs_uv.append(figure(width=800, height=800,
                          tools='save,reset', toolbar_location='right'))
    figs_uv[-1].add_tools(BoxZoomTool(match_aspect=True))
    
    common_props(figs_uv[-1], xlim=(-20,20), ylim=(-20,20))
    figs_uv[-1].hex_tile(q='u', r='v', source=source,
                         size=1, fill_color='color',
                         line_color='black', line_width=1, alpha=1.)
     
    figs_uv[-1].add_tools(HoverTool(tooltips=[('u/v', '@u/@v)'),]))

    # figs_uv[-1].text(xcart, ycart, text=['{}/{}'.format(u,v) for (u, v) in zip(u, v)],
    #                  text_baseline='middle', text_align='center',
    #                  text_font_size='7pt')
    tabs_uv.append(Panel(child=figs_uv[-1], title=str(il)))

    # (x,y) plots
    figs_xy.append(figure(width=800, height=800,
                          tools='save,reset', toolbar_location='right'))
    figs_xy[-1].add_tools(BoxZoomTool(match_aspect=True))
    figs_xy[-1].add_tools(HoverTool(tooltips=[('u/v', '@u/@v'),],))
    
    common_props(figs_xy[-1], xlim=(-13,13), ylim=(-13,13))
    figs_xy[-1].rect(x='u', y='v',
                     source=source,
                     width=1., height=1.,
                     width_units='data', height_units='data',
                     fill_color='color',
                     line_color='black',)
     
    #x, y = axial_to_cartesian(u, v, 1, 'pointytop') 
    # figs_xy[-1].text(u, v, text=['{}/{}'.format(u,v) for (u, v) in zip(u, v)],
    #                  text_baseline='middle', text_align='center',
    #                  text_font_size='7pt')
    tabs_xy.append(Panel(child=figs_xy[-1], title=str(il)))

lay = layout([[Tabs(tabs=tabs_uv),Tabs(tabs=tabs_xy)]])
save(lay)
