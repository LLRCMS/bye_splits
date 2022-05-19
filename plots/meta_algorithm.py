import os
import argparse
import sys; sys.path.append( os.environ['PWD'] )
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.io import output_file, show, save
from bokeh.models import (
    Range1d,
    ColumnDataSource,
    Whisker,
    BooleanFilter,
    CustomJS,
    CustomJSFilter,
    CDSView,
    Slider,
)

from airflow.airflow_dag import (
    optimization_kwargs as opt_kw,
    fill_path,
)

def stats_plotter():
    outcsv = fill_path(opt_kw['OptimizationCSVOut'], extension='csv')
    df = pd.read_csv(outcsv, sep=',', header=0)

    fig_opt = dict(width=600,
                   height=300,
                   #x_range=Range1d(-0.1, 1.1),
                   #y_range=Range1d(-0.05, 1.05),
                   tools="save",
                   x_axis_location='below',
                   x_axis_type='linear',
                   y_axis_type='linear')
    p1 = figure(title='Ratio of split clusters', **fig_opt)
    base_circle_opt = dict(x=df.hyperparameter,
                           size=4, line_width=4)
    base_line_opt = dict(x=df.hyperparameter,
                         line_width=3)
    cmssw_line_opt = dict(y=df.remrat2,
                            color='blue',
                            legend_label='CMSSW',
                            **base_line_opt)
    custom_circle_opt = dict(y=df.locrat2,
                             color='red',
                             legend_label='Custom',
                             **base_circle_opt)
    p1.line(**cmssw_line_opt)
    p1.circle(**custom_circle_opt)
    p1.legend.location = 'bottom_right'
    p1.toolbar.logo = None
    p1.xaxis.axis_label = 'Algorithm tunable parameter'

    return p1

def energy_resolution_plotter():
    assert len(opt_kw['FesAlgos']) == 1
    outooptimisationenres = fill_path(opt_kw['OptimizationEnResOut'], '')
    outooptimisationposres = fill_path(opt_kw['OptimizationPosResOut'], '')
    with pd.HDFStore(outooptimisationenres, mode='r') as storeEnRes, pd.HDFStore(outooptimisationposres, mode='r') as storePosRes:

        hyperparameters = storeEnRes[opt_kw['FesAlgos'][0] + '_meta' ].tolist()
        en_old, en_new = ([] for x in range(2))
        #std_old, std_new, mean_new, mean_old = ([] for x in range(4))
        for hp in hyperparameters:
            df = storeEnRes[ opt_kw['FesAlgos'][0] + '_data_' + str(hp).replace('.','p') ]

            en_old.append( np.sqrt(np.mean( df['enres_old']**2 )) / np.mean( df['enres_old'] ) )
            en_new.append( np.sqrt(np.mean( df['enres_new']**2 )) / np.mean( df['enres_new'] ) )
            # std_old.append( np.std( df['enres_old']) )
            # mean_old.append( np.mean( df['enres_old'] ) )
            # std_new.append( np.std( df['enres_new']) )
            # mean_new.append( np.mean( df['enres_new'] ) )
            
            hist_opt = dict(density=False, bins=50)
            hold, edgold = np.histogram(df['enres_old'], **hist_opt)
            hnew, edgnew = np.histogram(df['enres_new'], **hist_opt)

            line_centers = edgnew[1:]-(edgnew[1]-edgnew[0])/2
            repeated_parameter = [hp for _ in range(len(hnew))]
            if hp == hyperparameters[0]:
                enres_dict = dict(x=line_centers.tolist(),
                                  y=hnew.tolist(),
                                  hp=repeated_parameter)
            else:
                enres_dict['x'].extend(line_centers.tolist())
                enres_dict['y'].extend(hnew.tolist())
                enres_dict['hp'].extend(repeated_parameter)


        max_source = max( max(enres_dict['y']), max(hold) )
        enres_source = ColumnDataSource(data=enres_dict)
        
        callback = CustomJS(args=dict(s=enres_source), code="""
s.change.emit();
""")

        slider_d = dict(zip(np.arange(len(hyperparameters)),hyperparameters))
        from bokeh.models.formatters import  FuncTickFormatter
        fmt = FuncTickFormatter(code="""
var labels = {};
return labels[tick];
""".format(slider_d))

        slider = Slider(start=min(slider_d.keys()),
                        end=max(slider_d.keys()),
                        step=1, value=0, 
                        format=fmt)
# add to CustomJS: var labels = {};
# add to CustomJS: indices[i] = data[i] == labels[f];
# add to CustomJS: .format(slider_d)
        # slider = Slider(start=hyperparameters[0],
        #                 end=hyperparameters[-1],
        #                 value=hyperparameters[0],
        #                 step=.1, title='Tunable Parameter')
        slider.js_on_change('value', callback)

        filt = CustomJSFilter(args=dict(slider=slider), code="""
        var indices = new Array(source.get_length());
        var f = slider.value;
        var labels = {};
        const data = source.data['hp'];

        for (var i=0; i < source.get_length(); i++){{
            indices[i] = data[i] == labels[f];
        }}
        return indices;
        """.format(slider_d))
        view = CDSView(source=enres_source, filters=[filt])

        p = figure( width=600, height=300,
                    title='Energy Resolution: RecoPt/GenPt',
                    y_range=Range1d(-1, max_source+1),
                    tools='save,box_zoom,reset', y_axis_type='linear' )
        p.toolbar.logo = None
        virtualmin = 1e-4 #avoid log scale issues

        #quad_opt = dict(bottom=virtualmin, line_color='white', line_width=0)
        quad_opt = dict(line_width=3)
        p.step(x=edgold[1:]-(edgold[1]-edgold[0])/2,
               y=hold,
               color='blue', legend_label='CMSSW', **quad_opt)
        p.step(source=enres_source, view=view,
               x='x',
               y='y',
               color='red', legend_label='Custom', **quad_opt)

        p.legend.click_policy='hide'
        p.legend.location = 'top_left'

        if FLAGS.selection.startswith('above_eta_'):
            title_suf = ' (eta > ' + FLAGS.selection.split('above_eta_')[1] + ')'
        elif FLAGS.selection == 'splits_only':
            title_suf = '(split clusters only)'
        fig_summ = figure( width=600, height=300,
                           title='RMS {}'.format(title_suf),
                           tools='save,box_zoom,reset',
                           y_axis_type='linear',
                           #x_range=Range1d(-0.1, 1.1),
                           #y_range=Range1d(-0.05, 1.05),
                           )
        fig_summ.toolbar.logo = None
        fig_summ.xaxis.axis_label = 'Algorithm tunable parameter'

        # std_old = np.array(std_old)
        # std_new = np.array(std_new)
        # mean_old = np.array(mean_old)
        # mean_new = np.array(mean_new)
        points_opt = dict(x=hyperparameters)
        fig_summ.line(y=en_old, color='blue', legend_label='CMSSW',
                      line_width=3,
                      **points_opt) 
        fig_summ.circle(y=en_new, color='red', legend_label='Custom',
                        size=8,
                        **points_opt) 
        fig_summ.legend.click_policy='hide'
        fig_summ.legend.location = 'top_right'
        
        return fig_summ, p, slider
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--selection',
                        help='selection used to select cluster under study',
                        default='splits_only', type=str)
    FLAGS = parser.parse_args()

    suf = '_SEL_'
    if FLAGS.selection.startswith('above_eta_'):
        suf += FLAGS.selection
    elif FLAGS.selection == 'splits_only':
        suf += FLAGS.selection
    else:
        m = 'Selection {} is not supported.\n'.format(FLAGS.selection)
        m += 'Available options are: `splits_only` or `above_eta_ETAVALUE`'
        raise ValueError(m)

    this_file = os.path.basename(__file__).split('.')[0]
    output_file( os.path.join('out', this_file + suf + '.html') )
    
    stats_fig  = stats_plotter()
    summ_fig, enres_fig, slider = energy_resolution_plotter()

    ncols = 4
    lay_list = [[stats_fig, summ_fig], [slider], [enres_fig]]
    # for i,fig in enumerate(enres_figs):
    #     if i%4==0:
    #         lay_list.append([])
    #     lay_list[-1].append( fig )

    lay = layout(lay_list)
    save(lay) #if show_html else save(lay)
