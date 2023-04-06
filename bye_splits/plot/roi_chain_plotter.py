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
from bokeh import models as bmd

def common_props(fig):
    fig.output_backend = 'svg'
    fig.toolbar.logo = None
    fig.legend.click_policy='hide'
    fig.legend.location = 'top_left'
    fig.legend.label_text_font_size = '8pt'
    fig.min_border_bottom = 5
    fig.xaxis.visible = True

def resolution_plotter(df, pars, user):
    out = common.fill_path(os.path.join('/eos/user', user[0], user, 'www/L1/res'),
                           ext='html', **pars)
    bokeh.io.output_file(out)
    
    nbeta, nbphi, nben = (13 for _ in range(3))
    bins = {'eta': np.linspace(df.geneta.min(), df.geneta.max(), nbeta+1),
            'phi': np.linspace(df.genphi.min(), df.genphi.max(), nbphi+1),
            'en': np.linspace(df.genen.min(), df.genen.max(), nben+1)}

    def q1(x):
        return x.quantile(0.15865)

    def q3(x):
        return x.quantile(0.84135)

    avars = ['resroi', 'rescl']
    values = {x:df.groupby(pd.cut(df[x], bins[x])).agg(['median', q1, q3])[avars+'en']
              for x in bins.keys()}
    labels = {'eta': '\u03B7',
              'phi': '\u03C6',
              'en': 'Energy [GeV]'}
    
    figs = {x:None for x in bins.keys()}
    dvar = {'resroi': ('ROI Energy Resolution', 'ROI', 'blue'),
            'rescl': ('Cluster Enenergy Resolution', 'Clusters', 'red')}
    wshifts = {'eta': {'resroi': -0.01, 'rescl': 0.01},
               'phi': {'resroi': -0.1, 'rescl': 0.01},
               'en':  {'resroi': -10, 'rescl': 10}}

    for binvar in bins.keys():
        hshift = (bins[binvar][1]-bins[binvar][0])/2
        vshift = 0.1
        ymax = np.array(values[binvar]['resroi']['q3']).max()
        ymin = np.array(values[binvar]['rescl']['q1']).min()
        figs[binvar] = bokeh.plotting.figure(
            width=600, height=300, title='', tools='save',
            x_range=(bins[binvar][0], bins[binvar][-1]+4*hshift),
            #y_range=(ymin-vshift, ymax+vshift),
            y_range=(-0.52, 0.02),
            y_axis_type='linear'
        )

        for avar in avars:
            bincenters = (bins[binvar][:-1]+bins[binvar][1:])/2
            opt = dict(x=bincenters+wshifts[binvar][avar],
                       legend_label=dvar[avar][1],
                       color=dvar[avar][2])

            median = list(values[binvar][avar]['median'])
            q1 = np.array(values[binvar][avar]['q1'])
            q3 = np.array(values[binvar][avar]['q3'])
            medianopt = dict(y=median)

            # median
            figs[binvar].square(size=8, **medianopt, **opt)
            figs[binvar].line(line_width=1, **medianopt, **opt)
            figs[binvar].line(x=[bincenters[0]-hshift, bincenters[-1]+4*hshift], y=[0.,0.],
                              line_width=1, line_dash='dashed', color='gray')

            # quantiles
            source = bmd.ColumnDataSource(data=dict(base=bincenters + wshifts[binvar][avar],
                                                    q3=q3, q1=q1))
            quant = bmd.Whisker(base='base', upper='q3', lower='q1', source=source,
                                level='annotation', line_width=2, line_color=dvar[avar][2])
            quant.upper_head.size=10
            quant.lower_head.size=10
            quant.upper_head.line_color = dvar[avar][2]
            quant.lower_head.line_color = dvar[avar][2]
            figs[binvar].add_layout(quant)

            figs[binvar].xaxis.axis_label = labels[binvar]
            figs[binvar].yaxis.axis_label = r'$$E/E_{Gen}-1$$'
            common_props(figs[binvar])
            
    if pars.sel.startswith('above_eta_'):
        title_suf = ' (eta > ' + pars.selection.split('above_eta_')[1] + ')'
    elif pars.sel == 'splits_only':
        title_suf = '(split clusters only)'
    elif pars.sel == 'no_splits':
        title_suf = '(no split clusters)'

    ncols = 4
    sep = " <b>|</b> "
    text = sep.join(('<b>Selection: </b>{} ({} events)'.format(pars.sel, df.shape[0]),
                     '<b>Region: </b>{}'.format(pars.reg),
                     '<b>Seed window: </b>{}'.format(pars.seed_window),
                     '<b>Cluster algorithm: </b>{}'.format(pars.cluster_algo)))
    div = bokeh.models.Div(width=1000, height=20, text=text)
    lay_list = [[div], figs['eta'], figs['phi'], figs['en']]
    bokeh.io.save(bokeh.layouts.layout(lay_list))

    return None

def distribution_plotter(df, pars, user):
    out = common.fill_path(os.path.join('/eos/user', user[0], user, 'www/L1/dist'),
                           ext='html', **pars)
    bokeh.io.output_file(out)

    allfigs = []
    varpairs = (('roi', 'cl'),  ('resroi', 'rescl'))

    nbeta, nbphi, nben = (25 for _ in range(3))
    bins = {'eta': {'def': np.linspace(df.geneta.min(), df.geneta.max(), nbeta+1),
                    'res': np.linspace(-0.6, 0, nbeta+1)},
            'phi': {'def': np.linspace(df.genphi.min(), df.genphi.max(), nbphi+1),
                    'res': np.linspace(-0.6, 0, nbphi+1)},
            'en': {'def': np.linspace(df.genen.min(), df.genen.max(), nben+1),
                   'res': np.linspace(-0.6, 0, nben+1)}
            }
    labels = {'eta': {'def': '\u03B7', 'res': '\u03B7 resolution'},
              'phi': {'def': '\u03C6', 'res': '\u03C6 resolution'},
              'en':  {'def': 'Energy [GeV]', 'res': 'Energy resolution'}}
    dvar = {'eta': ('Eta distribution',),
            'phi': ('Phi distribution',),
            'en': ('Energy distribution',)}
    ranges = {'eta': {'def': (-2, 40), 'res': (-2, 100)},
              'phi': {'def': (-2, 35), 'res': (-2, 100)},
              'en':  {'def': (-2, 75), 'res': (-2, 100)}}
    
    for avars in varpairs:
        vstr = 'def' if avars==('roi', 'cl') else 'res'
        values = {x: {avars[0]: np.array(df.groupby(pd.cut(df[avars[0]+x], bins[x][vstr])).size()),
                      avars[1]: np.array(df.groupby(pd.cut(df[avars[1]+x], bins[x][vstr])).size())}
                  for x in bins.keys()}
        
        figs = {x:None for x in labels.keys()}
        values_d = {avars[0]: ('ROI', 'blue'),
                    avars[1]: ('Clusters', 'red')}
     
        for il in labels.keys():
            hshift = (bins[il][vstr][1]-bins[il][vstr][0])/2
            figs[il] = bokeh.plotting.figure(
                width=600, height=300, title='', tools='save',
                x_range=(bins[il][vstr][0], bins[il][vstr][-1]+4*hshift),
                #y_range=(ymin-vshift, ymax+vshift),
                y_range=ranges[il][vstr],
                y_axis_type='linear'
            )
            figs[il].xaxis.axis_label = labels[il][vstr]
            figs[il].yaxis.axis_label = 'Counts'
            
            bincenters = (bins[il][vstr][:-1]+bins[il][vstr][1:])/2
            for avar in avars:
                opt = dict(x=bincenters, y=values[il][avar],
                           legend_label=values_d[avar][0],
                           color=values_d[avar][1])
                figs[il].square(size=6, **opt)
                figs[il].line(line_width=1, **opt)
            common_props(figs[il])

        allfigs.append(figs)
            
    if pars.sel.startswith('above_eta_'):
        title_suf = ' (eta > ' + pars.selection.split('above_eta_')[1] + ')'
    elif pars.sel == 'splits_only':
        title_suf = '(split clusters only)'
    elif pars.sel == 'no_splits':
        title_suf = '(no split clusters)'

    ncols = 4
    sep = " <b>|</b> "
    text = sep.join(('<b>Selection: </b>{} ({} events)'.format(pars.sel, df.shape[0]),
                     '<b>Region: </b>{}'.format(pars.reg),
                     '<b>Seed window: </b>{}'.format(pars.seed_window),
                     '<b>Cluster algorithm: </b>{}'.format(pars.cluster_algo)))
    div = bokeh.models.Div(width=1000, height=20, text=text)
    lay_list = [[div],
                [allfigs[0]['en'],allfigs[1]['en']],
                allfigs[0]['phi'], allfigs[0]['eta']]
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
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
    window_u, window_v = cfg['seed_roi']['WindowUDim'], cfg['seed_roi']['WindowVDim']
    
    sep = " <b>|</b> "
    text = sep.join(('<b>Selection: </b>{}'.format(FLAGS.sel),
                     '<b>Region: </b>{}'.format(FLAGS.reg),
                     '<b>Seed u/v window: </b>{}/{}'.format(window_u,window_v),
                     '<b>Cluster algorithm: </b>{}'.format(FLAGS.cluster_algo)))
    div = bokeh.models.Div(width=1000, height=20, text=text)
    lay_list = [[div], [stats_fig], [slider], res_figs, res_ratios, summ_fig]

    lay = bokeh.layouts.layout(lay_list)
    bokeh.io.save(lay)


##Latex equations
# \Delta R \equiv  \sqrt{(\Delta \phi)^2+(\Delta \eta)^2}, \: \Delta \phi  \equiv \phi_{\text{Cluster}}-\phi_{\text{Gen}},  \: \Delta \eta  \equiv \eta_{\text{Cluster}}-\eta_{\text{Gen}}

#  \frac{E_{\text{Cluster}} - E_{\text{Gen}}}{E_{\text{Gen}}} < -0.35
