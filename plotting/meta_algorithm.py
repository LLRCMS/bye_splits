import os
import argparse
import sys; sys.path.append( os.environ['PWD'] )
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.layouts import layout
from bokeh.io import output_file, show, save, export_svg
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
    outcsv = fill_path(opt_kw['OptimizationCSVOut'], selection=FLAGS.selection, extension='csv')
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
                         line_width=2)
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

def resolution_plotter():
    assert len(opt_kw['FesAlgos']) == 1
    outooptimisationenres = fill_path(opt_kw['OptimizationEnResOut'], selection=FLAGS.selection)
    outooptimisationposres = fill_path(opt_kw['OptimizationPosResOut'], selection=FLAGS.selection)
    with pd.HDFStore(outooptimisationenres, mode='r') as storeEnRes, pd.HDFStore(outooptimisationposres, mode='r') as storePosRes:

        hyperparameters = storeEnRes[opt_kw['FesAlgos'][0] + '_meta' ].tolist()
        assert storePosRes[opt_kw['FesAlgos'][0] + '_meta' ].tolist() == hyperparameters

        res_dict = []
        en_old, en_new   = ([] for x in range(2))
        eta_old, eta_new = ([] for x in range(2))
        phi_old, phi_new = ([] for x in range(2))
        #std_old, std_new, mean_new, mean_old = ([] for x in range(4))
        for hp in hyperparameters:
            key = opt_kw['FesAlgos'][0]+'_data_'+str(hp).replace('.','p')
            df_en  = storeEnRes[key]
            df_pos = storePosRes[key]

            en_old.append(  np.std(df_en.enres_old)  )
            en_new.append(  np.std(df_en.enres_new)  )
            eta_old.append( np.std(df_pos.etares_old))
            eta_new.append( np.std(df_pos.etares_new) )
            phi_old.append( np.std(df_pos.phires_old))
            phi_new.append( np.std(df_pos.phires_new))
            adict_old = {0: en_old,
                         1: eta_old,
                         2: phi_old}
            adict_new = {0: en_new,
                         1: eta_new,
                         2: phi_new}

            mins = {0: min(min(df_en['enres_old']),  min(df_en['enres_new']))  -max(df_en['enres_old'])/15.,
                    1: min(min(df_pos['etares_old']),min(df_pos['etares_new']))-max(df_pos['etares_old'])/15.,
                    2: min(min(df_pos['phires_old']),min(df_pos['phires_new']))-max(df_pos['phires_old'])/15. 
                    }
            maxs = {0: max(max(df_en['enres_old']),  max(df_en['enres_new']))  +max(df_en['enres_new'])/15., 
                    1: max(max(df_pos['etares_old']),max(df_pos['etares_new']))+max(df_pos['etares_new'])/15.,  
                    2: max(max(df_pos['phires_old']),max(df_pos['phires_new']))+max(df_pos['phires_new'])/15.,  
                    }
            
            hist_opt = dict(density=False)
            hold, edgold = ([] for x in range(2))
            _tmp = np.histogram(df_en['enres_old'], bins=50, range=(mins[0],maxs[0]), **hist_opt)
            hold.append( _tmp[0] )
            edgold.append( _tmp[1] )
            for ii,x in enumerate(('etares_old', 'phires_old')):
                _tmp = np.histogram(df_pos[x], bins=50 if 'eta' in x else 150,
                                    range=(mins[ii+1],maxs[ii+1]), **hist_opt)
                hold.append( _tmp[0] )
                edgold.append( _tmp[1] )
            hnew, edgnew = ([] for x in range(2))
            _tmp = np.histogram(df_en['enres_new'], bins=50, range=(mins[0],maxs[0]), **hist_opt)
            hnew.append( _tmp[0] )
            edgnew.append( _tmp[1] )
            for ii,x in enumerate(('etares_new', 'phires_new')):
                _tmp = np.histogram(df_pos[x], bins=50 if 'eta' in x else 150,
                                    range=(mins[ii+1],maxs[ii+1]), **hist_opt)
                hnew.append( _tmp[0] )
                edgnew.append( _tmp[1] )

            for it in range(3):
                line_centers = edgnew[it][1:]-(edgnew[it][1]-edgnew[it][0])/2
                repeated_parameter = [hp for _ in range(len(hnew[it]))]
                if hp == hyperparameters[0]:
                    res_dict.append( dict(x=line_centers.tolist(),
                                          y=hnew[it].tolist(),
                                          hp=repeated_parameter) )
                else:
                    res_dict[it]['x'].extend(line_centers.tolist())
                    res_dict[it]['y'].extend(hnew[it].tolist())
                    res_dict[it]['hp'].extend(repeated_parameter)

        
        max_sources = [ max( max(res_dict[q]['y']), max(hold[q]) ) for q in range(3) ]
        sources = [ ColumnDataSource(data=res_dict[q]) for q in range(3) ]

        callback_str = """s.change.emit();"""
        callbacks  = [ CustomJS(args=dict(s=sources[q]),  code=callback_str)
                       for q in range(3) ]

        slider_d = dict(zip(np.arange(len(hyperparameters)),hyperparameters))
        from bokeh.models.formatters import FuncTickFormatter
        fmt = FuncTickFormatter(code="""
var labels = {};
return labels[tick];
""".format(slider_d))

        slider_opt = dict(start=min(slider_d.keys()),
                          end=max(slider_d.keys()),
                          step=1, value=0, 
                          format=fmt,
                          title='Iterative algorithm tunable parameter')
        slider = Slider(**slider_opt)
        # slider = Slider(start=hyperparameters[0],
        #                 end=hyperparameters[-1],
        #                 value=hyperparameters[0],
        #                 step=.1, title='Tunable Parameter')
        for it in range(3):
            slider.js_on_change('value', callbacks[it])

        filter_str = """
        var indices = new Array(source.get_length());
        var f = slider.value;
        var labels = {};
        const data = source.data['hp'];

        for (var i=0; i < source.get_length(); i++){{
            indices[i] = data[i] == labels[f];
        }}
        return indices;
        """.format(slider_d)
        filt  = [ CustomJSFilter(args=dict(slider=slider), code=filter_str)
                  for q in range(3) ]
        views  = [ CDSView(source=sources[q],  filters=[filt[q]])
                   for q in range(3) ]

        quad_opt = dict(line_width=1)
        cmssw_opt = dict(color='blue', legend_label='CMSSW', **quad_opt)
        custom_opt = dict(x='x',
                          y='y',
                          color='red', legend_label='Custom', **quad_opt)
        p = []
        title_d = {0: 'Energy Resolution: RecoPt/GenPt',
                   1: 'Eta Resolution: RecoEta - GenEta',
                   2: 'Phi Resolution: RecoPhi - GenPhi', }
        axis_label_d = {0: r'$$p_{T,\text{Reco}} / p_{T,\text{Gen}}$$',
                        1: r'$$\eta_{\text{Reco}} - \eta_{\text{Gen}}$$',
                        2: r'$$\phi_{\text{Reco}} - \phi_{\text{Gen}}$$', }
        for it in range(3):
            p.append( figure( width=600, height=300,
                              title=title_d[it],
                              y_range=Range1d(-1., max_sources[it]+1),
                              tools='save,box_zoom,reset',
                              y_axis_type='log' if it==3 else 'linear') )
            p[-1].toolbar.logo = None
            p[-1].step(x=edgold[it][1:]-(edgold[it][1]-edgold[it][0])/2,
                       y=hold[it], **cmssw_opt)
            p[-1].step(source=sources[it], view=views[it], **custom_opt)
            p[-1].legend.click_policy='hide'
            p[-1].legend.location = 'top_left'
            p[-1].xaxis.axis_label = axis_label_d[it]
            
        if FLAGS.selection.startswith('above_eta_'):
            title_suf = ' (eta > ' + FLAGS.selection.split('above_eta_')[1] + ')'
        elif FLAGS.selection == 'splits_only':
            title_suf = '(split clusters only)'

        figs_summ = []
        title_d = {0: 'Cluster Energy Resolution: standard deviations {}'.format(title_suf),
                   1: 'Cluster Eta Resolution: standard deviations {}'.format(title_suf),
                   2: 'Cluster Phi Resolution: standard deviations {}'.format(title_suf), }
        for it in range(3):
            figs_summ.append( figure( width=600, height=300,
                                      title=title_d[it],
                                      tools='save',
                                      y_axis_type='linear',
                                      #x_range=Range1d(-0.1, 1.1),
                                      #y_range=Range1d(-0.05, 1.05),
                                      ) )
            figs_summ[-1].toolbar.logo = None
            figs_summ[-1].xaxis.axis_label = 'Algorithm tunable parameter'

            points_opt = dict(x=hyperparameters)
            figs_summ[-1].line(y=adict_old[it], color='blue', legend_label='CMSSW',
                               line_width=1, **points_opt) 
            figs_summ[-1].circle(y=adict_new[it], color='red', legend_label='Custom',
                                 size=8,
                                 **points_opt) 
            figs_summ[-1].legend.click_policy='hide'
            figs_summ[-1].legend.location = 'top_right' if it==0 else 'bottom_right'
            figs_summ[-1].xaxis.axis_label = 'Algorithm tunable parameter'

            # figs_summ[-1].output_backend = "svg"
            # export_svg(figs_summ[-1], filename="plot.svg")
            # export_png(figs_summ[-1], filename="plot.png", height=300, width=300)

            
        return figs_summ, p, slider
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', '--selection',
                        help='selection used to select cluster under study',
                        default='splits_only', type=str)
    FLAGS = parser.parse_args()

    this_file = os.path.basename(__file__).split('.')[0]
    plot_name = fill_path(this_file,
                          selection=FLAGS.selection,
                          extension='html')

    output_file( plot_name )
    
    stats_fig  = stats_plotter()
    summ_fig, res_figs, slider = resolution_plotter()

    ncols = 4
    lay_list = [[stats_fig], [slider], res_figs, summ_fig ]

    lay = layout(lay_list)
    save(lay) #if show_html else save(lay)
