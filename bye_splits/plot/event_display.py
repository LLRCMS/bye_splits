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
from bokeh.models import Panel, Tabs, BoxZoomTool, Range1d
output_file('tmp.html')

import utils
from utils import params, common, parsing

flatten = lambda col : col.to_numpy().squeeze()

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
for il in layers:
    l_data = tc_data[(tc_data.layer==il) & (tc_data.zside==1) & (tc_data.subdet==1)][:]
    l_data = l_data[[var1, var2]].drop_duplicates().sort_values([var1, var2])

    u = flatten(l_data[var1])
    v = flatten(l_data[var2])
    v *= -1 #convert between CMSSW coordinate system and bokeh's
     
    figs_uv.append(figure(width=800, height=800,
                          tools='save,reset', toolbar_location='right'))
    figs_uv[-1].add_tools(BoxZoomTool(match_aspect=True))
    figs_uv[-1].output_backend = 'svg'
    figs_uv[-1].toolbar.logo = None
    figs_uv[-1].grid.visible = False
    figs_uv[-1].outline_line_color = None
    figs_uv[-1].x_range=Range1d(-20, 20)
    figs_uv[-1].y_range=Range1d(-20, 20)
    figs_uv[-1].xaxis.visible = False
    figs_uv[-1].yaxis.visible = False
     
    figs_uv[-1].hex_tile(u, v, size=1, fill_color=['firebrick']*len(u),
                         line_color='black', line_width=1, alpha=1.)
     
    x, y = axial_to_cartesian(u, v, 1, 'pointytop')
     
    figs_uv[-1].text(x, y, text=['{}/{}'.format(u,v) for (u, v) in zip(u, v)],
           text_baseline='middle', text_align='center',
           text_font_size='7pt')
    tabs_uv.append(Panel(child=figs_uv[-1], title=str(il)))
    
save(Tabs(tabs=tabs_uv))
