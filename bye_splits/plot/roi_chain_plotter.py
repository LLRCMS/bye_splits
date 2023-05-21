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
palette = bokeh.palettes.Dark2_5
from bokeh import models as bmd
import itertools  
import scipy

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

# def clopper_pearson(k, n, cl=0.95):
#     test = scipy.stats.binomtest(k, n, p=k/n, alternative='two-sided')
#     return test.proportion_ci(confidence_level=cl, method='exact')

def clopper_pearson(k, n, cl=0.95):
    """k is number of successes, n is number of trials"""
    c1 = 0 if k==0 else scipy.stats.beta.interval(cl, k, n-k+1)[0]
    c2 = 1 if k==n else scipy.stats.beta.interval(cl, k+1,n-k)[1]
    return c1, c2

def common_props(fig, title, legend=True):
    fig.output_backend = 'svg'
    fig.toolbar.logo = None
    if legend:
        fig.legend.click_policy='hide'
        fig.legend.location = 'bottom_right'
        fig.legend.label_text_font_size = '8pt'
    fig.min_border_bottom = 5
    fig.xaxis.visible = True
    fig.title=title
    fig.title.align = "left"
    fig.title.text_font_size = "15px"

    text_units = 'screen'
    labels = bmd.Label(x=4, y=310, x_units=text_units, y_units=text_units,
                       text="CMS Preliminary", text_font_size="13px",
                       text_font_style="italic")
    fig.add_layout(labels)

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
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
    extra = common.seed_extra_name(cfg)
    out = common.fill_path(os.path.join(folder, 'res'+extra), ext='html', **pars)
    bokeh.io.output_file(out)
    
    nbeta, nbphi, nben = (10 for _ in range(3))
    bins = {'eta': np.linspace(df.geneta.min(), df.geneta.max(), nbeta+1),
            'phi': np.linspace(df.genphi.min(), df.genphi.max(), nbphi+1),
            'en': np.linspace(df.genen.min(), df.genen.max(), nben+1)}

    def q1(x):
        return x.quantile(0.15865)

    def q3(x):
        return x.quantile(0.84135)

    around = {'eta': 2, 'phi': 2, 'en': 1}
    bincenters = {x: np.round((bins[x][:-1]+bins[x][1:])/2,around[x]).astype(str)
                  for x in bins.keys()}

    avars = ['clres', 'roiallres'] + ['roi' + str(t).replace('.','p') + 'res' for t in cfg['valid_cluster']['tcDeltaRthresh']]
    cuts = {x: pd.cut(df['gen'+x], bins[x], labels=bincenters[x], include_lowest=True) for x in bins.keys()}
    values = {x: df.groupby(cuts[x]).agg(['median', q1, q3, 'size'])[[k+'en' for k in avars]] for x in bins.keys()}

    labels = {'eta': '\u03B7',
              'phi': '\u03C6',
              'en': 'Energy [GeV]'}
    common_y_range = bmd.Range1d(-0.4, 0.12)
    violins = {}
    for bk in bins.keys():
        ibins = np.tile(cuts[bk].to_numpy(), len(avars)).astype(float)
        concat_list = [np.array(['Clusters' for _ in range(len(df['clresen']))]),
                       np.array(['All TCs'  for _ in range(len(df['roiallresen']))])]
        for it,t in enumerate(cfg['valid_cluster']['tcDeltaRthresh']):
            concat_list.append(np.array(['TCs (dR<{})'.format(t) for _ in range(len(df[avars[it+2]+'en']))]))
        group_ids = np.concatenate(concat_list, axis=None)

        vals_list = [df['clresen'], df['roiallresen']]
        for it,t in enumerate(cfg['valid_cluster']['tcDeltaRthresh']):
            vals_list.append(df['roi' + str(t).replace('.','p') + 'resen'])
        group_vals = pd.concat(vals_list)
        violins[bk] = hv.Violin((ibins.tolist(), group_ids, group_vals.to_numpy()), [bk, 'Group'])
        violins[bk] = violins[bk].opts(opts.Violin(height=400, width=1400,
                                                   violin_color=hv.dim('Group').str(),
                                                   show_legend=True,
                                                   box_color='black',
                                                   cut=0.2 # it seems a proxy for when to stop calculating the distribution tails
                                                   ), 
                                       clone=True,)
        violins[bk] = hv.render(violins[bk]) # convert to bokeh's format
        violins[bk].y_range = common_y_range

    figs = {x:None for x in bins.keys()}
    colors = itertools.cycle(palette) # create a color iterator

    dvar = {'clres': ('Cluster Enenergy Resolution', 'Clusters', next(colors)),
            'roiallres': ('TCs Energy Resolution', 'All TCs', next(colors))}
    for t in cfg['valid_cluster']['tcDeltaRthresh']:
        dvar.update({'roi' + str(t).replace('.','p') + 'res':
                     ('TCs Energy Resolution', 'TCs (dR<{})'.format(t), next(colors))})
    assert len(dvar.keys()) == len(avars)
        
    # calculate shifts along the x axis to avoid superposition of the lines
    horiz_shift = {'en': 10., 'eta': 0.01, 'phi': 0.1}
    start_x = {k:(-v*(len(avars)-1))/2 for k,v in horiz_shift.items()}
    next_x = lambda iteration, var: start_x[var] + iteration*horiz_shift[var]
    
    for bk in bins.keys():
        hshift = (bins[bk][1]-bins[bk][0])/2
        ymax = max([np.array(values[bk][x+'en']['q3']).max() for x in avars])
        ymin = min([np.array(values[bk][x+'en']['q1']).min() for x in avars])
        figs[bk] = bokeh.plotting.figure(
            width=1000, height=400, tools="box_zoom,reset,pan,save",
            x_range=(bins[bk][0], bins[bk][-1]+hshift),
            y_range=common_y_range,
            y_axis_type="linear"
        )

        for ivar, avar in enumerate(avars):
            bincenters = (bins[bk][:-1]+bins[bk][1:])/2
            opt = dict(x=bincenters+next_x(ivar,bk), legend_label=dvar[avar][1], color=dvar[avar][2])
            median = np.array(values[bk][avar+'en']['median'])
            q1     = np.array(values[bk][avar+'en']['q1'])
            q3     = np.array(values[bk][avar+'en']['q3'])
            medianopt = dict(y=median)

            # median
            figs[bk].circle(size=10, **medianopt, **opt)
            figs[bk].line(line_width=1, **medianopt, **opt)
            figs[bk].line(x=[bincenters[0]-hshift, bincenters[-1]+10*hshift], y=[0.,0.],
                              line_width=1, line_dash='dashed', color='gray')
            violins[bk].line(x=[0., len(avars)*nbeta+1.5*nbeta], y=[0.,0.],
                             line_width=1, line_dash='dashed', color='gray')
            violins[bk].xaxis.major_label_text_alpha = 0.
            violins[bk].xaxis.major_label_text_font_size = '0pt'
            violins[bk].xaxis.major_label_text_color = 'white'
            violins[bk].xaxis.major_tick_line_alpha = 0.
            violins[bk].xaxis.major_tick_line_color = 'white'

            # quantiles
            source = bmd.ColumnDataSource(data=dict(base=bincenters+next_x(ivar,bk),
                                                    q3=q3, q1=q1))
            quant = bmd.Whisker(base='base', upper='q3', lower='q1', source=source,
                                level='annotation', line_width=2, line_color=dvar[avar][2])
            quant.upper_head.size=10
            quant.lower_head.size=10
            quant.upper_head.line_color = dvar[avar][2]
            quant.lower_head.line_color = dvar[avar][2]
            figs[bk].add_layout(quant)

            figs[bk].xaxis.axis_label = labels[bk]
            violins[bk].xaxis.axis_label = labels[bk]
            figs[bk].yaxis.axis_label = r'$$E/E_{Gen}-1$$'
            violins[bk].yaxis.axis_label = r'$$E/E_{Gen}-1$$'
            
        common_props(figs[bk], title="Energy Response")
        common_props(violins[bk], title="Energy Response")
            
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
    div = bmd.Div(width=800, height=20, text=text)
    lay_list = [[div],
                [figs['en'], violins['en']],
                [figs['eta'], violins['eta']],
                [figs['phi'], violins['phi']]]
    bokeh.io.save(bokeh.layouts.layout(lay_list))

    return None

def seed_plotter(df, pars, user):
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
    extra = common.seed_extra_name(cfg)
    out = common.fill_path(os.path.join('/eos/user', user[0], user, 'www/L1/seed'+extra),
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
            'nrois': ('Number of ROIs', '#ROIs', 'green')}

    # calculate shifts along the x axis to avoid superposition of the lines
    horiz_shift = {'en': 10., 'eta': 0.01, 'phi': 0.1}
    start_x = {k:(-v*(len(avars)-1))/2 for k,v in horiz_shift.items()}
    next_x = lambda iteration, var: start_x[var] + iteration*horiz_shift[var]

    # efficiencies need a separate treatment
    avars.remove('has_seeds')
    avars.remove('less_seeds')

    # plot ROI and seed multiplicities
    for binvar in bins.keys():
        hshift = (bins[binvar][1]-bins[binvar][0])/2
        fcounts[binvar] = bokeh.plotting.figure(
            width=600, height=400, title='', tools='pan,save',
            x_range=(bins[binvar][0], bins[binvar][-1]+hshift),
            y_range=(0.9,1.6),
            y_axis_type='linear'
        )

        for ivar,avar in enumerate(avars):
            bincenters = (bins[binvar][:-1]+bins[binvar][1:])/2
            opt = dict(x=bincenters+next_x(ivar,binvar),
                       legend_label=dvar[avar][1],
                       color=dvar[avar][2])

            median = list(values[binvar][avar]['median'])
            mean = list(values[binvar][avar]['mean'])
            std = list(values[binvar][avar]['std'])
            q1 = np.array(values[binvar][avar]['q1'])
            q3 = np.array(values[binvar][avar]['q3'])
            mopt = dict(y=mean)

            # median
            fcounts[binvar].circle(size=10, **mopt, **opt)
            fcounts[binvar].line(line_width=1, **mopt, **opt)
            fcounts[binvar].line(x=[bincenters[0]-hshift, bincenters[-1]+hshift], y=[1.,1.],
                              line_width=1, line_dash='dashed', color='gray')

            # quantiles
            source = bmd.ColumnDataSource(data=dict(base=bincenters + next_x(ivar,binvar),
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
        common_props(fcounts[binvar], title="ROIs and Seeds multiplicities")

    # plot seeding efficiency
    for vvv in ('has_seeds', 'less_seeds'):
        for binvar in bins.keys():
            figname = binvar + '_' + vvv
            hshift = (bins[binvar][1]-bins[binvar][0])/2
            feff[figname] = bokeh.plotting.figure(
                width=600, height=400, title='', tools='pan,box_zoom,reset,save',
                x_range=(bins[binvar][0], bins[binvar][-1]+hshift),
                y_range=(0.6,1.4) if vvv=='has_seeds' else (-0.02,0.6),
                y_axis_type='linear'
            )
     
            bincenters = (bins[binvar][:-1]+bins[binvar][1:])/2
            opt = dict(x=bincenters+next_x(ivar,binvar), color=dvar[avar][2])
     
            k = np.array(values[binvar][vvv]['sum'])
            n = np.array(values[binvar][vvv]['size'])

            cp_interv = [clopper_pearson(kk, nn) for kk,nn in zip(k,n)]
            errlo = [cp[0] for cp in cp_interv]
            errup = [cp[1] for cp in cp_interv]
            
            eff = k / n
            feff[figname].circle(y=eff, size=10, **opt)
            feff[figname].line(line_width=1, **opt)
            feff[figname].line(x=[bincenters[0]-hshift, bincenters[-1]+4*hshift], y=[1.,1.],
                                    line_width=1, line_dash='dashed', color='gray')
     
            # errors
            source = bmd.ColumnDataSource(data=dict(base=bincenters + next_x(ivar,binvar),
                                                    errup=errup, errdown=errlo))
            quant = bmd.Whisker(base='base', upper='errup', lower='errdown', source=source,
                                level='annotation', line_width=2, line_color=dvar[avar][2])
            quant.upper_head.size=8
            quant.lower_head.size=8
            quant.upper_head.line_color = dvar[avar][2]
            quant.lower_head.line_color = dvar[avar][2]
            feff[figname].add_layout(quant)

            feff[figname].xaxis.axis_label = labels[binvar]
            feff[figname].yaxis.axis_label = ('Seeding Efficiency' if vvv=='has_seeds'
                                              else 'Fraction of events w/ #Seeds < #ROIs')
        common_props(feff[figname],
                     title="Seeding Efficiency" if vvv=="has_seeds" else "#Seeds < #ROIs", legend=False)
            
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
    div = bmd.Div(width=1000, height=20, text=text)
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
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
    extra = common.seed_extra_name(cfg)
    out = common.fill_path(os.path.join(folder, 'dist'+extra), ext='html', **pars)
    bokeh.io.output_file(out)

    allfigs = []
    varpairs = (('cl', 'roi0p1'),  ('clres', 'roi0p1res'))

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
    wshifts = {'eta': {'cl': 0.01, 'roi0p1': -0.01, 'clres': 0.002, 'roi0p1res': -0.002},
               'phi': {'cl': 0.01, 'roi0p1': -0.1, 'clres': 0.002, 'roi0p1res': -0.002, },
               'en':  {'cl': 10, 'roi0p1': -10, 'clres': 0.002, 'roi0p1res': -0.002}}
    dvar = {'eta': ('Eta distribution',),
            'phi': ('Phi distribution',),
            'en': ('Energy distribution',)}
    ranges = {'eta': {'def': (-2, 40), 'res': (-2, 170)},
              'phi': {'def': (-2, 135), 'res': (-2, 120)},
              'en':  {'def': (-2, 175), 'res': (-2, 120)}}
    
    for avars in varpairs:
        vstr = 'def' if avars==('cl', 'roi') else 'res'
        values = {x: {avars[0]: df.groupby(pd.cut(df[avars[0]+x], bins[x][vstr])).agg(['mean', 'std', 'size']),
                      avars[1]: df.groupby(pd.cut(df[avars[1]+x], bins[x][vstr])).agg(['mean', 'std', 'size'])}
                  for x in bins.keys()}

        figs = {x:None for x in labels.keys()}
        values_d = {avars[0]: ('Clusters', palette[0]),
                    avars[1]: ('TCs', palette[1])}

        for il in labels.keys():
            hshift = (bins[il][vstr][1]-bins[il][vstr][0])/2
            figs[il] = bokeh.plotting.figure(
                width=600, height=400, title='', tools='pan,box_zoom,reset,save',
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

                figs[il].circle(size=10, **opt)
                figs[il].line(line_width=1, **opt)
                
            common_props(figs[il], title="Distribution")

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
    div = bmd.Div(width=1000, height=20, text=text)
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
    div = bmd.Div(width=1000, height=20, text=text)
    lay_list = [[div], [stats_fig], [slider], res_figs, res_ratios, summ_fig]

    lay = bokeh.layouts.layout(lay_list)
    bokeh.io.save(lay)
