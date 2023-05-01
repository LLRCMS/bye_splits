# coding: utf-8

_all_ = [ 'stats_plotter', 'resolution_plotter' ]
    
import os
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import utils
from utils import common, params, parsing

import yaml
import argparse
import numpy as np
import pandas as pd
import bokeh
from bokeh import models as bmd

def common_props(fig, legend=True):
    fig.output_backend = 'svg'
    fig.toolbar.logo = None
    if legend:
        fig.legend.click_policy='hide'
        fig.legend.location = 'bottom_right'
        fig.legend.label_text_font_size = '8pt'
    fig.min_border_bottom = 5
    fig.xaxis.visible = True

def get_folder_plots():
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
    if cfg['cluster']['ROICylinder']:
        folder = 'CylinderOnly'
    else:
        folder = 'AllTCs'
    return folder

def resolution_plotter(df, pars, user):
    folder = get_folder_plots()
    folder = os.path.join('/eos/user', user[0], user, 'www/L1/', folder)
    common.create_dir(folder)
    out = common.fill_path(os.path.join(folder, 'res'), ext='html', **pars)
    bokeh.io.output_file(out)
    
    nbeta, nbphi, nben = (10 for _ in range(3))
    bins = {'eta': np.linspace(df.geneta.min(), df.geneta.max(), nbeta+1),
            'phi': np.linspace(df.genphi.min(), df.genphi.max(), nbphi+1),
            'en': np.linspace(df.genen.min(), df.genen.max(), nben+1)}

    def q1(x):
        return x.quantile(0.15865)

    def q3(x):
        return x.quantile(0.84135)

    avars = ['rescl']  #avars = ['resroi', 'rescl']
    values = {x:df.groupby(pd.cut(df['gen'+x], bins[x])).agg(['median', q1, q3, 'size'])[[k+'en' for k in avars]]
              for x in bins.keys()}
    labels = {'eta': '\u03B7',
              'phi': '\u03C6',
              'en': 'Energy [GeV]'}
    
    figs = {x:None for x in bins.keys()}
    dvar = {'resroi': ('TCs Energy Resolution', 'TCs', 'blue'),
            'rescl': ('Cluster Enenergy Resolution', 'Clusters', 'red')}
    wshifts = {'eta': {'resroi': -0.01, 'rescl': 0.01},
               'phi': {'resroi': -0.1, 'rescl': 0.01},
               'en':  {'resroi': -10, 'rescl': 10}}

    for bk in bins.keys():
        hshift = (bins[bk][1]-bins[bk][0])/2
        ymax = max([np.array(values[bk][x+'en']['q3']).max() for x in avars])
        ymin = min([np.array(values[bk][x+'en']['q1']).min() for x in avars])
        figs[bk] = bokeh.plotting.figure(
            width=600, height=300, title='', tools='box_zoom,reset,pan,save',
            x_range=(bins[bk][0], bins[bk][-1]+hshift),
            y_range=(-0.26, 0.02),
            y_axis_type='linear'
        )

        for avar in avars:
            bincenters = (bins[bk][:-1]+bins[bk][1:])/2
            opt = dict(x=bincenters+wshifts[bk][avar],
                       legend_label=dvar[avar][1],
                       color=dvar[avar][2])
            median = list(values[bk][avar+'en']['median'])
            q1 = np.array(values[bk][avar+'en']['q1'])
            q3 = np.array(values[bk][avar+'en']['q3'])
            medianopt = dict(y=median)

            # median
            figs[bk].square(size=8, **medianopt, **opt)
            figs[bk].line(line_width=1, **medianopt, **opt)
            figs[bk].line(x=[bincenters[0]-hshift, bincenters[-1]+hshift], y=[0.,0.],
                              line_width=1, line_dash='dashed', color='gray')

            # quantiles
            source = bmd.ColumnDataSource(data=dict(base=bincenters + wshifts[bk][avar],
                                                    q3=q3, q1=q1))
            quant = bmd.Whisker(base='base', upper='q3', lower='q1', source=source,
                                level='annotation', line_width=2, line_color=dvar[avar][2])
            quant.upper_head.size=10
            quant.lower_head.size=10
            quant.upper_head.line_color = dvar[avar][2]
            quant.lower_head.line_color = dvar[avar][2]
            figs[bk].add_layout(quant)

            figs[bk].xaxis.axis_label = labels[bk]
            figs[bk].yaxis.axis_label = r'$$E/E_{Gen}-1$$'
            common_props(figs[bk])
            
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

def seed_plotter(df, pars, user):
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)

    extra_name = '_hexdist' if cfg['seed_roi']['hexDist'] else ''
    out = common.fill_path(os.path.join('/eos/user', user[0], user,
                                        'www/L1/seed'+ extra_name),
                           ext='html', **pars)
    bokeh.io.output_file(out)

    # required to calculate the seeding efficiency
    df['has_seeds'] = (df.nseeds > 0).astype(int)
    df['less_seeds'] = (df.nseeds < df.nrois).astype(int)

    nbeta, nbphi, nben = (10 for _ in range(3))
    bins = {'eta': np.linspace(df.geneta.min(), df.geneta.max(), nbeta+1),
            'phi': np.linspace(df.genphi.min(), df.genphi.max(), nbphi+1),
            'en': np.linspace(df.genen.min(), df.genen.max(), nben+1)}

    def q1(x):
        return x.quantile(0.15865)

    def q3(x):
        return x.quantile(0.84135)

    avars = ['nseeds', 'nrois', 'has_seeds', 'less_seeds']
    values = {x:df.groupby(pd.cut(df['gen'+x], bins[x])).agg(['median', 'mean', 'std', q1, q3, 'sum', 'size'])[avars]
              for x in bins.keys()}
    labels = {'eta': '\u03B7',
              'phi': '\u03C6',
              'en': 'Energy [GeV]'}

    fcounts, feff  = ({x:None for x in bins.keys()} for _ in range(2))
    dvar = {'nseeds': ('Number of seeds', '#seeds', 'blue'),
            'nrois': ('Number of ROIs', '#ROIs', 'red')}
    wshifts = {'eta': [-0.01,0.,0.01],
               'phi': [-0.1,0.,0.01],
               'en': [-10,0,10]}

    # efficiencies need a separate treatment
    avars.remove('has_seeds')
    avars.remove('less_seeds')

    # plot ROI and seed multiplicities
    for binvar in bins.keys():
        hshift = (bins[binvar][1]-bins[binvar][0])/2
        fcounts[binvar] = bokeh.plotting.figure(
            width=600, height=300, title='', tools='pan,save',
            x_range=(bins[binvar][0], bins[binvar][-1]+hshift),
            y_range=(0.9,2.6),
            y_axis_type='linear'
        )

        for ivar,avar in enumerate(avars):
            bincenters = (bins[binvar][:-1]+bins[binvar][1:])/2
            opt = dict(x=bincenters+wshifts[binvar][ivar],
                       legend_label=dvar[avar][1],
                       color=dvar[avar][2])

            median = list(values[binvar][avar]['median'])
            mean = list(values[binvar][avar]['mean'])
            std = list(values[binvar][avar]['std'])
            q1 = np.array(values[binvar][avar]['q1'])
            q3 = np.array(values[binvar][avar]['q3'])
            mopt = dict(y=mean)

            # median
            fcounts[binvar].square(size=8, **mopt, **opt)
            fcounts[binvar].line(line_width=1, **mopt, **opt)
            fcounts[binvar].line(x=[bincenters[0]-hshift, bincenters[-1]+hshift], y=[1.,1.],
                              line_width=1, line_dash='dashed', color='gray')

            # quantiles
            source = bmd.ColumnDataSource(data=dict(base=bincenters + wshifts[binvar][ivar],
                                                    errup=mean+std/(2*np.sqrt(len(std))),
                                                    errdown=mean-std/(2*np.sqrt(len(std)))))
            quant = bmd.Whisker(base='base', upper='errup', lower='errdown', source=source,
                                level='annotation', line_width=2, line_color=dvar[avar][2])
            quant.upper_head.size=10
            quant.lower_head.size=10
            quant.upper_head.line_color = dvar[avar][2]
            quant.lower_head.line_color = dvar[avar][2]
            fcounts[binvar].add_layout(quant)

            fcounts[binvar].xaxis.axis_label = labels[binvar]
            fcounts[binvar].yaxis.axis_label = 'Average'
            common_props(fcounts[binvar])


    # plot seeding efficiency
    for vvv in ('has_seeds', 'less_seeds'):
        for binvar in bins.keys():
            figname = binvar + '_' + vvv
            hshift = (bins[binvar][1]-bins[binvar][0])/2
            feff[figname] = bokeh.plotting.figure(
                width=600, height=300, title='', tools='save',
                x_range=(bins[binvar][0], bins[binvar][-1]+hshift),
                y_range=(0.6,1.4) if vvv=='has_seeds' else (-0.1,1.1),
                y_axis_type='linear'
            )
     
            bincenters = (bins[binvar][:-1]+bins[binvar][1:])/2
            opt = dict(x=bincenters+wshifts[binvar][ivar], color=dvar[avar][2])
     
            k = np.array(values[binvar][vvv]['sum'])
            n = np.array(values[binvar][vvv]['size'])
            eff = k / n
            errup = [ee+ee*np.sqrt(1/kk+1/nn)/2  if kk!=0. else 0. for ee,kk,nn in zip(eff,k,n)]
            errlo = [ee-ee*np.sqrt(1/kk+1/nn)/2  if kk!=0. else 0. for ee,kk,nn in zip(eff,k,n)]
            feff[figname].square(y=eff, size=8, **opt)
            feff[figname].line(line_width=1, **opt)
            feff[figname].line(x=[bincenters[0]-hshift, bincenters[-1]+4*hshift], y=[1.,1.],
                                    line_width=1, line_dash='dashed', color='gray')
     
            # errors
            source = bmd.ColumnDataSource(data=dict(base=bincenters + wshifts[binvar][ivar],
                                                    errup=errup, errdown=errlo))
            quant = bmd.Whisker(base='base', upper='errup', lower='errdown', source=source,
                                level='annotation', line_width=2, line_color=dvar[avar][2])
            quant.upper_head.size=10
            quant.lower_head.size=10
            quant.upper_head.line_color = dvar[avar][2]
            quant.lower_head.line_color = dvar[avar][2]
            feff[figname].add_layout(quant)

            feff[figname].xaxis.axis_label = labels[binvar]
            feff[figname].yaxis.axis_label = ('Seeding Efficiency' if vvv=='has_seeds'
                                              else 'Fraction of events w/ #Seeds < #ROIs')
            common_props(feff[figname], legend=False)
            
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
                     '<b>Seed window: </b>{}'.format(pars.seed_window)))
    div = bokeh.models.Div(width=1000, height=20, text=text)
    lay_list = [[div],
                [fcounts['en'], feff['en_has_seeds'], feff['en_less_seeds']],
                [fcounts['eta'], feff['eta_has_seeds'], feff['eta_less_seeds']],
                [fcounts['phi'], feff['phi_has_seeds'], feff['phi_less_seeds']]]
    bokeh.io.save(bokeh.layouts.layout(lay_list))

    return None

def distribution_plotter(df, pars, user):
    folder = get_folder_plots()
    folder = os.path.join('/eos/user', user[0], user, 'www/L1/', folder)
    common.create_dir(folder)
    out = common.fill_path(os.path.join(folder, 'dist'), ext='html', **pars)
    bokeh.io.output_file(out)

    allfigs = []
    varpairs = (('roi', 'cl'),  ('resroi', 'rescl'))

    nbeta, nbphi, nben = 200, 200, 30
    bins = {'eta': {'def': np.linspace(df.geneta.min(), df.geneta.max(), nbeta+1),
                    'res': np.linspace(-0.6, 0, nbeta+1)},
            'phi': {'def': np.linspace(df.genphi.min(), df.genphi.max(), nbphi+1),
                    'res': np.linspace(-0.6, 0, nbphi+1)},
            'en': {'def': np.linspace(df.genen.min(), df.genen.max(), nben+1),
                   'res': np.linspace(-0.6, 0, nben+1)}
            }
    labels = {'eta': {'def': '\u03B7', 'res': '\u03B7 resolution'},
              'phi': {'def': '\u03C6', 'res': '\u03C6 resolution'},
              'en':  {'def': 'Energy [GeV]', 'res': r'$$E/E_{Gen}-1$$'}}
    wshifts = {'eta': {'roi': -0.01, 'cl': 0.01, 'resroi': -0.002, 'rescl': 0.002},
               'phi': {'roi': -0.1, 'cl': 0.01, 'resroi': -0.002, 'rescl': 0.002},
               'en':  {'roi': -10, 'cl': 10, 'resroi': -0.002, 'rescl': 0.002}}
    dvar = {'eta': ('Eta distribution',),
            'phi': ('Phi distribution',),
            'en': ('Energy distribution',)}
    ranges = {'eta': {'def': (-2, 40), 'res': (-2, 170)},
              'phi': {'def': (-2, 135), 'res': (-2, 120)},
              'en':  {'def': (-2, 175), 'res': (-2, 120)}}
    
    for avars in varpairs:
        vstr = 'def' if avars==('roi', 'cl') else 'res'
        values = {x: {avars[0]: df.groupby(pd.cut(df[avars[0]+x], bins[x][vstr])).agg(['mean', 'std', 'size']),
                      avars[1]: df.groupby(pd.cut(df[avars[1]+x], bins[x][vstr])).agg(['mean', 'std', 'size'])}
                  for x in bins.keys()}

        figs = {x:None for x in labels.keys()}
        values_d = {avars[0]: ('TCs', 'blue'),
                    avars[1]: ('Clusters', 'red')}

        for il in labels.keys():
            hshift = (bins[il][vstr][1]-bins[il][vstr][0])/2
            figs[il] = bokeh.plotting.figure(
                width=600, height=300, title='', tools='pan,box_zoom,reset,save',
                x_range=(bins[il][vstr][0], bins[il][vstr][-1]+4*hshift),
                # y_range=ranges[il][vstr],
                y_axis_type='linear'
            )
            figs[il].xaxis.axis_label = labels[il][vstr]
            figs[il].yaxis.axis_label = 'Counts'
            
            bincenters = (bins[il][vstr][:-1]+bins[il][vstr][1:])/2
            for avar in avars:
                size = list(values[il][avar][avar+il]['size'])
                mean = list(values[il][avar][avar+il]['mean'])
                std = list(values[il][avar][avar+il]['std'])
                opt = dict(x=bincenters + wshifts[il][avar],
                           y=size,
                           legend_label=values_d[avar][0],
                           color=values_d[avar][1])

                # errors
                source = bmd.ColumnDataSource(data=dict(base=bincenters + wshifts[il][avar],
                                                        errup=size+np.sqrt(size),
                                                        errdown=size-np.sqrt(size)))
                quant = bmd.Whisker(base='base', upper='errup', lower='errdown', source=source,
                                    level='annotation', line_width=2, line_color=values_d[avar][1])
                quant.upper_head.size=10
                quant.lower_head.size=10
                quant.upper_head.line_color = values_d[avar][1]
                quant.lower_head.line_color = values_d[avar][1]
                figs[il].add_layout(quant)

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
                [allfigs[0]['phi'],allfigs[1]['phi']],
                [allfigs[0]['eta'],allfigs[1]['eta']]]
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
