import os
import sys; sys.path.append( os.environ['PWD'] )
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.io import output_file, show, save
from bokeh.models import (
    Range1d,
    ColumnDataSource,
    Whisker
)

from airflow.airflow_dag import (
    optimization_kwargs,
)

def stats_plotter():
    df = pd.read_csv(os.path.join('data', 'stats.csv'), sep=',', header=0)

    fig_opt = dict(width=600,
                   height=300,
                   x_range=Range1d(-0.1, 1.1),
                   #y_range=Range1d(-0.05, 1.05),
                   tools="save",
                   x_axis_location='below',
                   x_axis_type='linear',
                   y_axis_type='linear')
    p1 = figure(title='Ratio of splitted clusters', **fig_opt)
    base_circle_opt = dict(x=df.hyperparameter,
                           size=4, line_width=4)
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
        en_old, en_new = ([] for x in range(2))
        #std_old, std_new, mean_new, mean_old = ([] for x in range(4))
        for hp in hyperparameters:
            df = storeEnRes[ optimization_kwargs['FesAlgos'][0] + '_data_' + str(hp).replace('.','p') ]

            en_old.append( np.sqrt(np.mean( df['enres_old']**2 )) / np.mean( df['enres_old'] ) )
            en_new.append( np.sqrt(np.mean( df['enres_new']**2 )) / np.mean( df['enres_new'] ) )
            # std_old.append( np.std( df['enres_old']) )
            # mean_old.append( np.mean( df['enres_old'] ) )
            # std_new.append( np.std( df['enres_new']) )
            # mean_new.append( np.mean( df['enres_new'] ) )
            
            hist_opt = dict(density=False, bins=int(len(df)/50))
            hold, edgold = np.histogram(df['enres_old'], **hist_opt)
            hnew, edgnew = np.histogram(df['enres_new'], **hist_opt)
             
            p = figure( width=400, height=250, title='Energy Resolution (algo parameter = {})'.format(hp),
                       tools='save,box_zoom,reset', y_axis_type='linear')
            p.toolbar.logo = None
        
            virtualmin = 1e-4 #avoid log scale issues
            quad_opt = dict(bottom=virtualmin, line_color='white', line_width=0)
            p.quad(top=hold, left=edgold[:-1], right=edgold[1:],
                   fill_color='blue', legend_label='CMSSW', alpha=1., **quad_opt)
            p.quad(top=hnew, left=edgnew[:-1], right=edgnew[1:],
                   fill_color='red', legend_label='Custom', alpha=0.5, **quad_opt)
            p.legend.click_policy='hide'
            figures.append(p)

        fig_summ = figure( width=600, height=300, title='Energy Resolution Summary (eta > 2.7 )'.format(hp),
                           tools='save,box_zoom,reset', y_axis_type='linear',
                           x_range=Range1d(-0.1, 1.1),
                           #y_range=Range1d(-0.05, 1.05),
                           )
        fig_summ.toolbar.logo = None
        fig_summ.xaxis.axis_label = 'Algorithm tunable parameter'

        # std_old = np.array(std_old)
        # std_new = np.array(std_new)
        # mean_old = np.array(mean_old)
        # mean_new = np.array(mean_new)
        points_opt = dict(x=hyperparameters, size=8)
        fig_summ.circle(y=en_old, color='blue', legend_label='CMSSW', alpha=1., **points_opt) 
        fig_summ.circle(y=en_new, color='red', legend_label='Custom', alpha=1., **points_opt) 
        fig_summ.legend.click_policy='hide'

        # source_old = ColumnDataSource(data=dict(base=hyperparameters,
        #                                         lower=abs(mean_old-std_old/2),
        #                                         upper=abs(mean_old+std_old/2)))
        # source_new = ColumnDataSource(data=dict(base=hyperparameters,
        #                                         lower=abs(mean_new-std_new/2),
        #                                         upper=abs(mean_new+std_new/2)))

        # fig_summ.add_layout(
        #     Whisker(source=source_old, base="base", upper="upper", lower="lower",
        #             line_color='blue')
        # )
        # fig_summ.add_layout(
        #     Whisker(source=source_new, base="base", upper="upper", lower="lower",
        #             line_color='red')
        # )
        
        return fig_summ, figures
    
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
    summ_fig, enres_figs = energy_resolution_plotter()

    ncols = 4
    lay_list = [[stats_fig, summ_fig]]
    for i,fig in enumerate(enres_figs):
        if i%4==0:
            lay_list.append([])
        lay_list[-1].append( fig )

    lay = layout(lay_list)
    show(lay) #if show_html else save(lay)
