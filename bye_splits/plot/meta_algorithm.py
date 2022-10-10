# coding: utf-8

_all_ = [ 'stats_plotter', 'resolution_plotter' ]
    
import os
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import utils
from utils import common, params, parsing

import argparse
import numpy as np
import pandas as pd

import bokeh
from bokeh import plotting
from bokeh import io
from bokeh import models

def stats_plotter(pars, names_d):
    for par in pars:
        incsv = common.fill_path(params.opt_kw['OptCSVOut'], ipar=par, ext='csv', **names_d)

        df_tmp = pd.read_csv(incsv, sep=',', header=0)
        if par == pars[0]:
            df = df_tmp[:]
        else:
            df = pd.concat((df,df_tmp))

    fig_opt = dict(width=600,
                   height=300,
                   #x_range=models.Range1d(-0.1, 1.1),
                   #y_range=models.Range1d(-0.05, 1.05),
                   tools="save",
                   x_axis_location='below',
                   x_axis_type='linear',
                   y_axis_type='linear')
    p1 = plotting.figure(title='Ratio of split clusters', **fig_opt)
    base_circle_opt = dict(x=df.ipar, size=4, line_width=4)
    base_line_opt = dict(x=df.ipar,
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

def resolution_plotter(pars, names_d):
    assert len(params.opt_kw['FesAlgos']) == 1

    # aggregating data produced with different values of the iterative algo parameter
    # in_en_tot  = common.fill_path(params.opt_kw['OptimizationEnResOut'], selection=FLAGS.sel)
    # in_pos_tot = common.fill_path(params.opt_kw['OptimizationPosResOut'], selection=FLAGS.sel)
    # with pd.HDFStore(in_en_tot, mode='w') as storeEnRes, pd.HDFStore(in_pos_tot, mode='w') as storePosRes:

    #     key = params.opt_kw['FesAlgos'][0] + '_data'
    #     for par in pars:
    #         in_en  = common.fill_path(pars.opt_kw['OptimizationEnResOut'],
    #                            param=par, selection=FLAGS.sel)
    #         in_pos = common.fill_path(pars.opt_kw['OptimizationPosResOut'],
    #                            param=par, selection=FLAGS.sel)
    #         with pd.HDFStore(in_en, mode='w') as tmpEnRes, pd.HDFStore(in_pos, mode='w') as tmpPosRes:
    #             storeEnRes [key] = tmpEnRes[key]
    #             storePosRes[key] = tmpPosRes[key]

    # # produce plots using the aggregated data
    # with pd.HDFStore(in_en_tot, mode='r') as storeEnRes, pd.HDFStore(in_pos_tot, mode='r') as storePosRes:

    res_dict = []
    en_old, en_new   = ([] for x in range(2))
    eta_old, eta_new = ([] for x in range(2))
    phi_old, phi_new = ([] for x in range(2))

    # First loop over all the datasets to calculate plot optimal ranges
    mins = {0:  1e10, 1:  1e10, 2:  1e10 }
    maxs = {0: -1e10, 1: -1e10, 2: -1e10 }
    for par in pars:
        nd = dict(ipar=par, **names_d)
        in_en  = common.fill_path(params.opt_kw['OptEnResOut'], **nd)
        in_pos = common.fill_path(params.opt_kw['OptPosResOut'], **nd)

        with pd.HDFStore(in_en, mode='r') as tmpEnRes, pd.HDFStore(in_pos, mode='r') as tmpPosRes:            
            key = params.opt_kw['FesAlgos'][0]+'_data'
            df_en  = tmpEnRes[key]
            df_pos = tmpPosRes[key]

            shift = {0: 0, 1: 0, 2: 0}
            if par == pars[-1]:
                scale = 15.
                shift = { 0: max(df_en.enres_old)  / scale,
                          1: max(df_pos.etares_old) / scale,
                          2: max(df_pos.phires_old) / scale }
            
            mins[0] = min(mins[0], min(min(df_en.enres_old),    min(df_en.enres_new)))   - shift[0]
            mins[1] = min(mins[1], min(min(df_pos.etares_old),  min(df_pos.etares_new))) - shift[1]
            mins[2] = min(mins[2], min(min(df_pos.phires_old),  min(df_pos.phires_new))) - shift[2]
            maxs[0] = max(maxs[0], max(max(df_en.enres_old),    max(df_en.enres_new)))   + shift[0]
            maxs[1] = max(maxs[1], max(max(df_pos.etares_old),  max(df_pos.etares_new))) + shift[1]
            maxs[2] = max(maxs[2], max(max(df_pos.phires_old),  max(df_pos.phires_new))) + shift[2]

    # Second loop to plot
    for par in pars:
        nd = dict(ipar=par, **names_d)
        in_en  = common.fill_path(params.opt_kw['OptEnResOut'], **nd)
        in_pos = common.fill_path(params.opt_kw['OptPosResOut'], **nd)                               
        with pd.HDFStore(in_en, mode='r') as tmpEnRes, pd.HDFStore(in_pos, mode='r') as tmpPosRes:            
            #key = params.opt_kw['FesAlgos'][0]+'_data_'+str(hp).replace('.','p')
            key = params.opt_kw['FesAlgos'][0]+'_data'

            df_en  = tmpEnRes[key]
            df_pos = tmpPosRes[key]
 
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
        
            hist_opt = dict(density=False)
            hold, edgold = ([] for x in range(2))
            _tmp = np.histogram(df_en['enres_old'], bins=50, range=(mins[0],maxs[0]), **hist_opt)
            hold.append( _tmp[0] )
            edgold.append( _tmp[1] )
            for ii,x in enumerate(('etares_old', 'phires_old')):
                _tmp = np.histogram(df_pos[x], bins=50,
                                    range=(mins[ii+1],maxs[ii+1]), **hist_opt)
                hold.append( _tmp[0] )
                edgold.append( _tmp[1] )
            hnew, edgnew = ([] for x in range(2))
            _tmp = np.histogram(df_en['enres_new'], bins=50, range=(mins[0],maxs[0]), **hist_opt)
            hnew.append( _tmp[0] )
            edgnew.append( _tmp[1] )
            for ii,x in enumerate(('etares_new', 'phires_new')):
                _tmp = np.histogram(df_pos[x], bins=50,
                                    range=(mins[ii+1],maxs[ii+1]), **hist_opt)
                hnew.append( _tmp[0] )
                edgnew.append( _tmp[1] )
 
            for it in range(3):
                line_centers = edgnew[it][1:]-(edgnew[it][1]-edgnew[it][0])/2
                repeated_parameter = [par for _ in range(len(hnew[it]))]

                if par == pars[0]:
                    res_dict.append( dict(x=line_centers.tolist(),
                                          y=hnew[it].tolist(),
                                          par=repeated_parameter) )
                else:
                    res_dict[it]['x'].extend(line_centers.tolist())
                    res_dict[it]['y'].extend(hnew[it].tolist())
                    res_dict[it]['par'].extend(repeated_parameter)

    max_sources = [ max( max(res_dict[q]['y']), max(hold[q]) ) for q in range(3) ]
    sources = [ models.ColumnDataSource(data=res_dict[q]) for q in range(3) ]
    
    callback_str = """s.change.emit();"""
    callbacks  = [ models.CustomJS(args=dict(s=sources[q]),  code=callback_str)
                   for q in range(3) ]

    slider_d = dict(zip(np.arange(len(pars)),pars))
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
    slider = models.Slider(**slider_opt)
    for it in range(3):
        slider.js_on_change('value', callbacks[it])

    filter_str = """
                 var indices = new Array(source.get_length());
                 var f = slider.value;
                 var labels = {};
                 const data = source.data['par'];
                  
                 for (var i=0; i < source.get_length(); i++){{
                 indices[i] = data[i] == labels[f];
                 }}
                 return indices;
                 """.format(slider_d)
    filt  = [ models.CustomJSFilter(args=dict(slider=slider), code=filter_str)
              for q in range(3) ]
    views  = [ models.CDSView(source=sources[q],  filters=[filt[q]])
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
        p.append( plotting.figure( width=600, height=300,
                                  title=title_d[it],
                                  y_range=models.Range1d(-1., max_sources[it]+1),
                                  tools='save,box_zoom,reset',
                                  y_axis_type='log' if it==3 else 'linear') )
        p[-1].toolbar.logo = None
        p[-1].step(x=edgold[it][1:]-(edgold[it][1]-edgold[it][0])/2,
                   y=hold[it], **cmssw_opt)
        p[-1].step(source=sources[it], view=views[it], **custom_opt)
        p[-1].legend.click_policy='hide'
        p[-1].legend.location = 'top_right' if it==2 else 'top_left'
        p[-1].xaxis.axis_label = axis_label_d[it]
            
    if FLAGS.sel.startswith('above_eta_'):
        title_suf = ' (eta > ' + FLAGS.selection.split('above_eta_')[1] + ')'
    elif FLAGS.sel == 'splits_only':
        title_suf = '(split clusters only)'

    figs_summ = []
    title_d = {0: 'Cluster Energy Resolution: standard deviations {}'.format(title_suf),
               1: 'Cluster Eta Resolution: standard deviations {}'.format(title_suf),
               2: 'Cluster Phi Resolution: standard deviations {}'.format(title_suf), }
    for it in range(3):
        figs_summ.append( plotting.figure( width=600, height=300,
                                          title=title_d[it],
                                          tools='save',
                                          y_axis_type='linear',
                                          #x_range=models.Range1d(-0.1, 1.1),
                                          #y_range=models.Range1d(-0.05, 1.05),
                                          ) )
        figs_summ[-1].toolbar.logo = None
        figs_summ[-1].xaxis.axis_label = 'Algorithm tunable parameter'

        points_opt = dict(x=pars)
        figs_summ[-1].line(y=adict_old[it], color='blue', legend_label='CMSSW',
                           line_width=1, **points_opt) 
        figs_summ[-1].circle(y=adict_new[it], color='red', legend_label='Custom',
                             size=8,
                             **points_opt) 
        figs_summ[-1].legend.click_policy='hide'
        figs_summ[-1].legend.location = 'top_right' if it==0 else 'bottom_right'
        figs_summ[-1].xaxis.axis_label = 'Algorithm tunable parameter'
        
        # figs_summ[-1].output_backend = "svg"
        # io.export_svg(figs_summ[-1], filename="plot.svg")
        # io.export_png(figs_summ[-1], filename="plot.png", height=300, width=300)
    return figs_summ, p, slider
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meta-plotter.')
    parser.add_argument('-m', '--params',
                        help='Iterative parameters.',
                        default=[0.0,1.0], nargs='+', type=float)
    parsing.add_parameters(parser, meta=True)
    FLAGS = parser.parse_args()

    this_file = os.path.basename(__file__).split('.')[0]
    names_d = dict(sel=FLAGS.sel,
                   reg=FLAGS.reg,
                   seed_window=FLAGS.seed_window,
                   smooth_kernel=FLAGS.smooth_kernel,
                   cluster_algo=FLAGS.cluster_algo)
    plot_name = common.fill_path(this_file, ext='html', **names_d)

    io.output_file( plot_name )
    
    stats_fig  = stats_plotter(pars=FLAGS.params, names_d=names_d)
    summ_fig, res_figs, slider = resolution_plotter(pars=FLAGS.params, names_d=names_d)

    ncols = 4
    lay_list = [[stats_fig], [slider], res_figs, summ_fig]

    lay = bokeh.layouts.layout(lay_list)
    io.save(lay)


##Latex equations
# \Delta R \equiv  \sqrt{(\Delta \phi)^2+(\Delta \eta)^2}, \: \Delta \phi  \equiv \phi_{\text{Cluster}}-\phi_{\text{Gen}},  \: \Delta \eta  \equiv \eta_{\text{Cluster}}-\eta_{\text{Gen}}

#  \frac{E_{\text{Cluster}} - E_{\text{Gen}}}{E_{\text{Gen}}} < -0.35
