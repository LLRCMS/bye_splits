# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

from functools import partial
import argparse
import numpy as np
import pandas as pd
import uproot as up
import awkward as ak

#from bokeh.io import output_file, save
#output_file('tmp.html')
from bokeh.plotting import figure, curdoc
from bokeh.util.hex import axial_to_cartesian
from bokeh.models import (
    Div,
    BoxZoomTool,
    Range1d,
    ColumnDataSource,
    HoverTool,
    TextInput,
    Tabs,
    Slider,
    CustomJS,
    CustomJSFilter,
    CDSView,
    WheelZoomTool,
    )
from bokeh.layouts import layout
from bokeh.settings import settings

df = pd.DataFrame({'x': [0,1,2,3], 'y': [0,1,2,3]})
source = ColumnDataSource(data=df)
doc = curdoc()
width, height = int(1600/3), 400
d = 1
d4 = d
cos30 = np.sqrt(3)/2
sin30 = 1/2
coords = {'UL': (lambda v: cos30 * d4 * v,
                 lambda u,v: sin30 * d4 * (2*(u-1)-v)),
          'UR': (lambda v: 4*cos30 + cos30 * d4 * (v-4),
                 lambda u,v: 2 + sin30 * d4 * (2*(u-4)-(v-4))),
          'B':  (lambda v: cos30 * d4 * v,
                 lambda u,v: sin30 * d4 * (2*u-v))
          } #up-right, up-left and bottom

us = np.array([1,2,3,4,2,3,4,5,3,4,5,6,4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7,0,1,2,3,0,1,2,3,0,1,2,3,0,1,3])
vs = np.array([0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,3,4,5,6,2,3,4,5,1,2,3,4,0,1,3])

masks = {'UL': (((us>=1) & (us<=4) & (vs==0)) |
                ((us>=2) & (us<=5) & (vs==1)) |
                ((us>=3) & (us<=6) & (vs==2)) |
                ((us>=4) & (us<=7) & (vs==3))),
         'UR': (us>=4) & (us<=7) & (vs>=4) & (vs<=7),
         'B':  (vs>=us) & (us<=3),
         }

x0, x1, x2, x3 = ({} for _ in range(4))
y0, y1, y2, y3 = ({} for _ in range(4))
xaxis, yaxis = ({} for _ in range(2))
for key,val in masks.items():
    x0.update({key: coords[key][0](vs[masks[key]])})
    x1.update({key: x0[key][:] + cos30})
    if key in ('UL', 'UR'):
        x2.update({key: x1[key][:]})
        x3.update({key: x0[key][:]})
    else:
        x2.update({key: x1[key][:] + cos30})
        x3.update({key: x1[key][:]})

    y0.update({key: coords[key][1](us[masks[key]],vs[masks[key]])})
    if key in ('UR', 'B'):
        y1.update({key: y0[key][:] - sin30})
    else:
        y1.update({key: y0[key][:] + sin30})
    if key in ('B'):
        y2.update({key: y0[key][:]})
    else:
        y2.update({key: y1[key][:] + d})
    if key in ('UL', 'UR'):
        y3.update({key: y0[key][:] + d})
    else:
        y3.update({key: y0[key][:] + sin30})

    xaxis.update({key: np.stack([x0[key],x1[key],x2[key],x3[key]],
                                axis=1).reshape((-1,1,1,4))})
    yaxis.update({key: np.stack([y0[key],y1[key],y2[key],y3[key]],
                                axis=1).reshape((-1,1,1,4))})

polyg_opt = dict(line_color='black', line_width=3)
p_cells = figure(width=width, height=height,
                 tools='save,reset', toolbar_location='right',
                 output_backend='webgl')
p_cells.add_tools(BoxZoomTool(match_aspect=True))
for key in masks.keys():
    p_cells.multi_polygons(color='white',
                           xs=xaxis[key].tolist(),
                           ys=yaxis[key].tolist(),
                           **polyg_opt)

lay = layout([[p_cells],])
doc.add_root(lay)
