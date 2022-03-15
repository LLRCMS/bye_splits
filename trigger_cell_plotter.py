import numpy as np

from airflow.airflow_dag import base_kwargs
    
from bokeh.io import output_file, show
from bokeh.layouts import layout
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LogColorMapper, LogTicker,
                          LinearColorMapper, BasicTicker,
                          PrintfTickFormatter,
                          Range1d,
                          Panel, Tabs)
from bokeh.plotting import figure
from bokeh.palettes import viridis as _palette

def plot_trigger_cells(rzslices):
    mypalette = _palette(50)
    title = r'{} vs {} bins'.format(base_kwargs['NbinsPhi'],
                                    base_kwargs['NbinsRz'])

    outbins = [ np.digitize(rzslice, bins=base_kwargs['PhiBinEdges'])
                for rzslice in rzslices ]
    source = ColumnDataSource({'outdata': outdata,
                               'outbins': outbins})
    bincounts = np.bincounts(outbins)
    mapper = LinearColorMapper(palette=mypalette,
                               low=bincounts.min(),
                               high=bincounts.max())

    p = figure(width=1800, height=600, title=title,
               x_range=Range1d(base_kwargs['MinPhi'], base_kwargs['MaxPhi']),
               y_range=Range1d(base_kwargs['MinROverZ'], base_kwargs['MaxROverZ']),
               tools="hover,box_select,box_zoom,undo,redo,reset,save",
               x_axis_location='below',
               x_axis_type='linear', y_axis_type='linear',
               )

    p.rect( x=outdata, y=tcNames.RoverZ,
            source=source,
            width=binDistPhi, height=binDistRz,
            width_units='data', height_units='data',
            line_color='black', fill_color=transform(tcNames.nhits, mapper)
           )

        color_bar = ColorBar(color_mapper=mapper,
                             ticker= ( LogTicker(desired_num_ticks=len(mypalette))
                                       if FLAGS.log else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) ),
                             formatter=PrintfTickFormatter(format="%d")
                             )
        p.add_layout(color_bar, 'right')

        set_figure_props(p, phiBinCenters, rzBinCenters)

        p.hover.tooltips = [
            ("#hits", "@{nhits}"),
            ("min(eta)", "@{min_eta}"),
            ("max(eta)", "@{max_eta}"),
        ]

        tc_backgrounds.append( p )
