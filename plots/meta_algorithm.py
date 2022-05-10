import os
import sys; sys.path.append( os.environ['PWD'] )
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.io import output_file, show, save
from bokeh.models import (
    Range1d,
)
from airflow.airflow_dag import (
    optimization_kwargs,
)

def stats_plotter():
    df = pd.read_csv(os.path.join('data', 'stats.csv'), sep=',', header=0)

    fig_opt = dict(width=600,
                   height=300,
                   x_range=Range1d(-0.1, 1.1),
                   y_range=Range1d(-0.05, 1.05),
                   tools="save",
                   x_axis_location='below',
                   x_axis_type='linear',
                   y_axis_type='linear')
    p1 = figure(title='Ratio of splitted clusters', **fig_opt)
    base_circle_opt = dict(x=df.hyperparameter,
                           size=6, line_width=4)
    cmssw_circle_opt = dict(y=df.remrat2,
                            color='blue',
                            legend_label='CMSSW',
                            **base_circle_opt)
    custom_circle_opt = dict(y=df.locrat2,
                             color='red',
                             legend_label='Custom algorithm',
                             **base_circle_opt)
    p1.circle(**cmssw_circle_opt)
    p1.circle(**custom_circle_opt)
    p1.legend.location = 'bottom_right'
    p1.toolbar.logo = None
    p1.xaxis.axis_label = 'Algorithm tunable parameter'

    return p1

def energy_resolution_plotter():

    assert len(optimization_kwargs['FesAlgos']) == 1
    with pd.HDFStore(optimization_kwargs['OptimizationEnResOut'], mode='r') as storeEnRes:

        hyperparameters = storeEnRes[optimization_kwargs['FesAlgos'][0] + '_meta' ]
        figures = []
        for hp in hyperparameters:
            df = storeEnRes[ optimization_kwargs['FesAlgos'][0] + '_data_' + str(hp).replace('.','p') ]

            hist_opt = dict(density=False, bins=30)
            hold, edgold = np.histogram(df['enres_old'], **hist_opt)
            hnew, edgnew = np.histogram(df['enres_new'], **hist_opt)
             
            p = figure( width=400, height=250, title='Energy Resolution (hyperparameter={})'.format(hp),
                       y_axis_type="linear")
            p.toolbar.logo = None
        
            virtualmin = 1e-4 #avoid log scale issues
            quad_opt = dict(bottom=virtualmin, line_color="white", alpha=0.6)
            p.quad(top=hold, left=edgold[:-1], right=edgold[1:],
                   fill_color='blue', legend_label='CMSSW', **quad_opt)
            p.quad(top=hnew, left=edgnew[:-1], right=edgnew[1:],
                   fill_color='red', legend_label='Custom', **quad_opt)
            p.legend.click_policy='hide'
            figures.append(p)

        return figures
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--ledges', help='layer edges (if -1 is added the full range is also included)', default=[0,28], nargs='+', type=int)
    # parser.add_argument('--pos_endcap', help='Use only the positive endcap.',
    #                     default=True, type=bool)
    # parser.add_argument('--hcal', help='Consider HCAL instead of default ECAL.', action='store_true')
    # parser.add_argument('-l', '--log', help='use color log scale', action='store_true')

    # FLAGS = parser.parse_args()

    output_file( os.path.join('out', 'stats_collection.html') )
    
    stats_fig  = stats_plotter()
    enres_figs = energy_resolution_plotter()

    ncols = 4
    lay_list = [[stats_fig]]
    for i,fig in enumerate(enres_figs):
        if i%4==0:
            lay_list.append([])
        lay_list[-1].append( fig )

    lay = layout(lay_list)
    show(lay) #if show_html else save(lay)
