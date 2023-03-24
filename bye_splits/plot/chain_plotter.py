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

def stats_plotter(pars, names_d):
    for par in pars:
        incsv = common.fill_path(params.opt_kw['OptCSVOut'], ipar=par, ext='csv', **names_d)

        df_tmp = pd.read_csv(incsv, sep=',', header=0)
        if par == pars[0]:
            df = df_tmp[:]
        else:
            df = pd.concat((df,df_tmp))

    fig_opt = dict(width=400,
                   height=300,
                   #x_range=bokeh.models.Range1d(-0.1, 1.1),
                   #y_range=bokeh.models.Range1d(-0.05, 1.05),
                   tools="save",
                   x_axis_location='below',
                   x_axis_type='linear',
                   y_axis_type='linear')
    p1 = bokeh.plotting.figure(title='Ratio of split clusters', **fig_opt)
    p1.output_backend = 'svg'
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

def resolution_plotter(df, pars, user):
    out = common.fill_path(os.path.join('/eos/user', user[0], user, 'www/L1/reco'),
                           ext='html', **pars)
    bokeh.io.output_file(out)

    avars = ('enres', 'etares', 'phires')
    for avar in avars:
        df[avar + '_diff']  = df[avar + '_new'] - df[avar + '_old']

    # Calculate plot optimal ranges
    mins = {0:  1e10, 1:  1e10, 2:  1e10 }
    maxs = {0: -1e10, 1: -1e10, 2: -1e10 }

    scale = 15.
    shift = {0: max(df.enres_old)  / scale,
             1: max(df.etares_old) / scale,
             2: max(df.phires_old) / scale }
            
    mins[0] = min(mins[0], min(min(df.enres_old),   min(df.enres_new)))  - shift[0]
    mins[1] = min(mins[1], min(min(df.etares_old),  min(df.etares_new))) - shift[1]
    mins[2] = min(mins[2], min(min(df.phires_old),  min(df.phires_new))) - shift[2]
    maxs[0] = max(maxs[0], max(max(df.enres_old),   max(df.enres_new)))  + shift[0]
    maxs[1] = max(maxs[1], max(max(df.etares_old),  max(df.etares_new))) + shift[1]
    maxs[2] = max(maxs[2], max(max(df.phires_old),  max(df.phires_new))) + shift[2]

    # Plot        
    hist_opt = dict(density=False, bins=55)
    hold, edgold = ([] for x in range(2))
    hnew, edgnew = ([] for x in range(2))
    hratio, ratup, ratlow = ([] for x in range(3))
    for ix,x in enumerate(avars):
        tmp1 = np.histogram(df[x + '_old'], range=(mins[ix],maxs[ix]), **hist_opt)
        hold.append(tmp1[0])
        edgold.append(tmp1[1])
        tmp2 = np.histogram(df[x + '_new'], range=(mins[ix],maxs[ix]), **hist_opt)
        hnew.append(tmp2[0])
        edgnew.append(tmp2[1])

        outzeros = np.zeros_like(tmp1[0], dtype=np.float64)
        rat = np.divide(tmp1[0], tmp2[0], out=np.zeros_like(tmp1[0], dtype=np.float64), where=tmp2[0]!=0)
        seg = rat*np.sqrt(np.divide(1., tmp1[0], out=outzeros, where=tmp1[0]!=0) +
                          np.divide(1., tmp2[0], out=outzeros, where=tmp2[0]!=0))

        hratio.append(rat)
        ratup.append(rat+seg/2)
        ratlow.append(rat-seg/2)

    centers = []
    for it in range(len(avars)):
        centers.append(edgold[it][1:]-(edgold[it][1]-edgold[it][0])/2) # could have used 'edgnew' instead

    figs, figs_ratio = ([] for _ in range(2))
    title_d = {0: 'Energy Resolution: RecoPt/GenPt',
               1: 'Eta Resolution: RecoEta - GenEta',
               2: 'Phi Resolution: RecoPhi - GenPhi', }
    axis_label_d = {0: r'$$p_{T,\text{Reco}} / p_{T,\text{Gen}}$$',
                    1: r'$$\eta_{\text{Reco}} - \eta_{\text{Gen}}$$',
                    2: r'$$\phi_{\text{Reco}} - \phi_{\text{Gen}}$$', }

    quad_opt = dict(line_width=1)
    for it in range(len(avars)):
        figs.append(bokeh.plotting.figure(width=400, height=300,
                                          title=title_d[it],
                                          #y_range=bokeh.models.Range1d(-1., max_source+1),
                                          tools='save,box_zoom,reset',
                                          y_axis_type='log' if it==3 else 'linear'))
        figs[-1].step(x=centers[it], y=hold[it],
                      color='blue', legend_label='CMSSW', **quad_opt)
        figs[-1].step(x=centers[it], y=hnew[it],
                      color='red', legend_label='Custom', **quad_opt)
        figs[-1].output_backend = 'svg'
        figs[-1].toolbar.logo = None
        figs[-1].legend.click_policy='hide'
        figs[-1].legend.location = 'top_right' if it==1 else 'top_left'
        figs[-1].min_border_bottom = 5
        figs[-1].xaxis.visible = False

        figs_ratio.append(bokeh.plotting.figure(width=400, height=200,
                                                y_range=bokeh.models.Range1d(0.61, 1.39),
                                                tools='save,box_zoom,reset',
                                                y_axis_type='linear'))
        figs_ratio[-1].circle(x=centers[it], y=hratio[it],
                              color='gray', size=4, legend_label='CMSSW/Custom')
        figs_ratio[-1].segment(x0=centers[it], x1=centers[it],
                               y0=ratlow[it], y1=ratup[it],
                               color='gray', line_width=2, legend_label='CMSSW/Custom')
        figs_ratio[-1].output_backend = 'svg'
        figs_ratio[-1].toolbar.logo = None
        figs_ratio[-1].xaxis.axis_label = axis_label_d[it]
        #figs_ratio[-1].yaxis.axis_label = 'Custom / CMSSW'
        figs_ratio[-1].min_border_top = 5

    if pars.sel.startswith('above_eta_'):
        title_suf = ' (eta > ' + pars.selection.split('above_eta_')[1] + ')'
    elif pars.sel == 'splits_only':
        title_suf = '(split clusters only)'
    elif pars.sel == 'no_splits':
        title_suf = '(no split clusters)'

    ncols = 4
    sep = " <b>|</b> "
    text = sep.join(('<b>Selection: </b>{}'.format(pars.sel),
                     '<b>Region: </b>{}'.format(pars.reg),
                     '<b>Seed window: </b>{}'.format(pars.seed_window),
                     '<b>Smooth kernel: </b>{}'.format(pars.smooth_kernel),
                     '<b>Cluster algorithm: </b>{}'.format(pars.cluster_algo)))
    div = bokeh.models.Div(width=1000, height=20, text=text)
    lay_list = [[div], figs, figs_ratio]
    bokeh.io.save(bokeh.layouts.layout(lay_list))

    return None
    
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

    bokeh.io.output_file( plot_name )
    
    stats_fig  = stats_plotter(pars=FLAGS.params, names_d=names_d)
    res_figs, res_ratios, slider = resolution_plotter(pars=FLAGS.params, names_d=names_d)

    ncols = 4
    sep = " <b>|</b> "
    text = sep.join(('<b>Selection: </b>{}'.format(FLAGS.sel),
                     '<b>Region: </b>{}'.format(FLAGS.reg),
                     '<b>Seed window: </b>{}'.format(FLAGS.seed_window),
                     '<b>Smooth kernel: </b>{}'.format(FLAGS.smooth_kernel),
                     '<b>Cluster algorithm: </b>{}'.format(FLAGS.cluster_algo)))
    div = bokeh.models.Div(width=1000, height=20, text=text)
    lay_list = [[div], [stats_fig], [slider], res_figs, res_ratios, summ_fig]

    lay = bokeh.layouts.layout(lay_list)
    bokeh.io.save(lay)
