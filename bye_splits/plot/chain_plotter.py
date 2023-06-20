# coding: utf-8

_all_ = [ 'ChainPlotter' ]
    
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
import itertools
import scipy
        
import bokeh
from bokeh import models as bmd

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

class ChainPlotter:
    """
    Manages all plotting within the reconstruction chains.
    """
    def __init__(self, chain_mode='default', user='bfontana', tag=''):
        assert chain_mode in ('default', 'cs', 'both')
        self.mode = chain_mode
        self.user = user
        self.tag = tag
        self.uc = {'eta': '\u03B7', 'phi': '\u03D5'}
        self.info = {
            'eta': {'title_1d': self.uc['eta']+' response', 'label_1d': r'$$\eta-\eta_{Gen}$$',
                    'title_2d': self.uc['eta']+' response', 'label_2d': '|'+self.uc['eta']+'|',
                    'nbins_2d': 10, 'nbins_1d': 80, 'round': 2},
            'phi': {'title_1d': self.uc['phi']+' response', 'label_1d': r'$$\phi-\phi_{Gen}$$',
                    'title_2d': self.uc['eta']+' response', 'label_2d': self.uc['phi'],
                    'nbins_2d': 10, 'nbins_1d': 50, 'round': 2},
            'en':  {'title_1d': 'Energy response', 'label_1d': r'$$E/E_{Gen}-1$$',
                    'title_2d': 'Energy response', 'label_2d': 'Energy [GeV]',
                    'nbins_2d': 10, 'nbins_1d': 50, 'round': 1},
            'pt':  {'title_1d': 'pT response', 'label_1d': r'$$p_{T}/p_{T,Gen}-1$$',
                    'title_2d': 'pT response', 'label_2d': 'pT [GeV]',
                    'nbins_2d': 10, 'nbins_1d': 50, 'round': 1},
        }

        self._set_plot_attributes()

        self._q1 = lambda x : x.quantile(0.15865)
        self._q3 = lambda x : x.quantile(0.84135)
        self._nanlen = lambda x : np.count_nonzero(np.isnan(x))

        self.bincenters = lambda bins, r=9 : np.round((bins[:-1]+bins[1:])/2, r)
        
        with open(params.CfgPath, 'r') as afile:
            self.cfg = yaml.safe_load(afile)

        self.out_dir = os.path.join('/eos/user', self.user[0], self.user, 'www/L1')

    def _clopper_pearson(self, k, n, cl=0.95, method='beta'):
        """k is number of successes, n is number of trials"""
        if method == 'beta':
            c1 = 0 if k==0 else scipy.stats.beta.interval(cl, k, n-k+1)[0]
            c2 = 1 if k==n else scipy.stats.beta.interval(cl, k+1,n-k)[1]
            return c1, c2
        elif method == 'binom':
            test = scipy.stats.binomtest(k, n, p=k/n, alternative='two-sided')
            return test.proportion_ci(confidence_level=cl, method='exact')

    def _output(self, s, pars, folder=''):
        """Defines a common output naming scheme, starting with string `s`."""
        name = s + common.seed_extra_name(self.cfg) + '_CHAIN' + self.mode
        path = os.path.join(self.out_dir, folder, name)
        path = common.fill_path(base_path=path, ext='html', **pars)
        bokeh.io.output_file(path)

    def _display(self, layout, pars, nevents, no_cluster=False):
        """Assigns and saves bokeh plots into a html layout."""
        sep = " <b>|</b> "
        text = sep.join(('<b>Selection: </b>{} ({} events)'.format(pars.sel, nevents),
                         '<b>Region: </b>{}'.format(pars.reg),
                         '<b>Seed window: </b>{}'.format(pars.seed_window)))
        if not no_cluster:
            text += sep + '<b>Cluster algorithm: </b>{}'.format(pars.cluster_algo)
        div = bmd.Div(width=1000, height=20, text=text)
        lay = [[div]] + layout
        bokeh.io.save(bokeh.layouts.layout(lay))

    def _set_plot_attributes(self):
        """Set plotting related attributes"""
        self.fig_height = 400
        self.fig_tools = 'pan,box_zoom,reset,save'
        self.whisker_head_size = 8
        self.horiz_shift = {'en'   : 10.,      'eta': 0.01,  'phi'   : 0.1,   'pt': 1.,
                            'enres': 0.000, 'etares': 0.000, 'phires': 0.000, 'ptres': 1.}

        self.palette = bokeh.palettes.Category10[10] #bokeh.palettes.Dark2_5
        

    def _x_shift(self, iteration, var, nclasses):
        """calculate shifts along the x axis to avoid superposition of the lines"""
        start_x = {k:(-v*(nclasses-1))/2 for k,v in self.horiz_shift.items()}
        return start_x[var] + iteration*self.horiz_shift[var]
        
    def _set_fig_common_attributes(self, fig, title, legend=True, location='bottom_right', show_grid=True):
        fig.output_backend = 'svg'
        fig.toolbar.logo = None
        if legend:
            fig.legend.click_policy='hide'
            fig.legend.location = location
            fig.legend.label_text_font_size = '15px'
        fig.min_border_bottom = 5

        # title
        fig.add_layout(bmd.Title(
            text="CMS Simulation", text_font_size="18px", align="left",
            text_font_style="italic"), 'above')

        # grid
        for g in ('xgrid', 'ygrid'):
            grid = getattr(fig, g)
            grid.visible = show_grid
            grid.grid_line_color = "black"
            grid.grid_line_alpha = 0.25
            grid.grid_line_width = 0.6
            grid.band_fill_color = "white"

        # axis
        for ax in ('xaxis', 'yaxis'):
            axis = getattr(fig, ax)
            axis.visible = True
            axis.major_label_text_font_style = "normal"
            axis.axis_label_text_font_style = "italic"
            axis.axis_label_text_font_size = "18px"
            axis.axis_label_text_font_size = "16px"
            axis.major_label_text_font_size = "16px"
            axis.axis_line_width = 1
            axis.major_tick_line_width = 1.5
            axis.minor_tick_line_width = 1
            axis.major_label_text_align = "center"
            axis.axis_label_text_color = "black"
            #axis.axis_label_text_outline_color = "black"
            axis.major_label_text_color = "black"
            #axis.major_label_text_outline_color = "black"
            axis.axis_line_color = "black"
            axis.major_tick_line_color = "black"
                
    def _get_bins(self, df):
        """Creates a dictionary with bins based on generated quantities."""
        gen = {'eta': df.geneta, 'phi': df.genphi, 'en': df.genen, 'pt': df.genpt}
        bins_1d = {'pt' : np.linspace(-3,     3,  self.info['pt']['nbins_1d']+1),
                   'en' : np.linspace(-0.2,   0.04,  self.info['en']['nbins_1d']+1),
                   'eta': np.linspace(-0.009, 0.009, self.info['eta']['nbins_1d']+1),
                   'phi': np.linspace(-0.009, 0.009,  self.info['phi']['nbins_1d']+1)}
        bins_2d = {v: np.linspace(gen[v].min(), gen[v].max(), self.info[v]['nbins_2d']+1)
                   for v in self.info.keys()}
        return bins_1d, bins_2d

    def seed_plotter(self, df, pars):
        self.df_seed = df
        self._output('seed_'+self.tag, pars, folder='SeedROIStudies')
     
        # required to calculate the seeding efficiency
        if self.mode == 'both':
            for suf in ('_def', '_roi'):
                nans  = df['nseeds'+suf].isna() #NaN's cannot be converted to a boolean!
                df['has_seeds'+suf]  = (df['nseeds'+suf] > 0).astype(int)
                df['has_seeds'+suf][nans] = np.nan
                df['less_seeds'+suf] = (df['nseeds'+suf] < df['nrois'+suf]).astype(int)
                df['less_seeds'+suf][nans] = np.nan
        else:
            df['has_seeds'] = (df.nseeds > 0).astype(int)
            df['less_seeds'] = (df.nseeds < df.nrois).astype(int)
     
        _, bins = self._get_bins(df)

        if self.mode == 'both':
            avars = []
            for suf in ('_def', '_roi'):
                avars.extend(['nseeds' + suf, 'nrois' + suf, 'has_seeds' + suf, 'less_seeds' + suf])
                #avars.extend(['nseeds' + suf, 'has_seeds' + suf, 'less_seeds' + suf])
        else:
            avars = ['nseeds', 'nrois', 'has_seeds', 'less_seeds']
        aggr_quantities = ['median', 'mean', 'std', self._q1, self._q3, 'sum', 'size', self._nanlen]
        values = {x: df.groupby(pd.cut(df['gen'+x], bins[x]))
                  .agg(aggr_quantities)
                  .rename(columns={'<lambda_0>': 'q1', '<lambda_1>': 'q3', '<lambda_2>': 'nanlen'})[avars]
                  for x in bins.keys()}
        fcounts, feff  = ({x:None for x in bins.keys()} for _ in range(2))

        if self.mode == 'both':
            leglab = {}
            _leglab = {'_def': 'R/z,'+self.uc['phi'], '_roi': 'CS'}
            for suf in _leglab.keys():
                leglab.update({'nseeds'+suf:    '#seeds ('+_leglab[suf]+')',
                               'nrois' +suf:    '#CS regions ('+_leglab[suf]+')',
                               'has_seeds'+suf:  _leglab[suf],
                               'less_seeds'+suf: _leglab[suf],
                               })
        else:
            leglab = {'nseeds': '#seeds', 'nrois':  '#CS regions'}
     
        # efficiencies need a separate treatment
        if self.mode == 'both':
            for suf in ('_def', '_roi'):
                avars.remove('has_seeds'+suf)
                avars.remove('less_seeds'+suf)
        else:
            avars.remove('has_seeds')
            avars.remove('less_seeds')
     
        # plot ROI and seed multiplicities
        average_numbers_y_range = (0.98,1.25) if self.cfg['seed_roi']=='NoROItcOut' else (0.98,1.5)
        for binvar in bins.keys():
            hshift = (bins[binvar][1]-bins[binvar][0])/2
            fcounts[binvar] = bokeh.plotting.figure(
                width=600, height=self.fig_height, title='', tools=self.fig_tools,
                x_range=(bins[binvar][0], bins[binvar][-1]+hshift),
                y_range=average_numbers_y_range,
                y_axis_type='linear'
            )
     
            for ivar,avar in enumerate(avars):
                # skip plotting number of ROIs for the default chain
                bincenters = (bins[binvar][:-1]+bins[binvar][1:])/2
                opt = dict(x=bincenters+self._x_shift(ivar,binvar,len(avars)),
                           legend_label=leglab[avar], color=self.palette[ivar])
     
                #median = list(values[binvar][avar]['median'])
                mean   = list(values[binvar][avar]['mean'])
                std    = list(values[binvar][avar]['std'])
                q1     = np.array(values[binvar][avar]['q1'])
                q3     = np.array(values[binvar][avar]['q3'])
                mopt = dict(y=mean)
     
                # median
                fcounts[binvar].circle(size=10, **mopt, **opt)
                fcounts[binvar].line(line_width=1, **mopt, **opt)
                fcounts[binvar].line(x=[bincenters[0]-hshift, bincenters[-1]+hshift], y=[1.,1.],
                                     line_width=1, line_dash='dashed', color='gray')
     
                # quantiles
                source = bmd.ColumnDataSource(
                    data=dict(base=bincenters+self._x_shift(ivar,binvar,len(avars)),
                              errup=mean+std/(2*np.sqrt(len(std))),
                              errdown=mean-std/(2*np.sqrt(len(std)))))
                quant = bmd.Whisker(base='base', upper='errup', lower='errdown', source=source,
                                    level='annotation', line_width=2, line_color=self.palette[ivar])
                quant.upper_head.size=self.whisker_head_size
                quant.lower_head.size=self.whisker_head_size
                quant.upper_head.line_color = self.palette[ivar]
                quant.lower_head.line_color = self.palette[ivar]
                fcounts[binvar].add_layout(quant)
     
                fcounts[binvar].xaxis.axis_label = self.info[binvar]['label_2d']
                fcounts[binvar].yaxis.axis_label = 'Average of the number of seeds and ROIs'
            self._set_fig_common_attributes(fcounts[binvar], title="ROIs and Seeds multiplicities",
                                            location='top_right')
     
        # plot seeding efficiency
        for ivar,vvv in enumerate(('has_seeds', 'less_seeds')):
            roi_waste_y_range = ((0.8,1.2) if 'has_seeds' in vvv else (-0.02,0.2)
                                 if self.cfg['seed_roi']['InputName']=='NoROItcOut' else (-0.02,0.35))
            for binvar in bins.keys():
                figname = binvar + '_' + vvv
                hshift = (bins[binvar][1]-bins[binvar][0])/2
                feff[figname] = bokeh.plotting.figure(
                    width=600, height=self.fig_height, title='', tools=self.fig_tools,
                    x_range=(bins[binvar][0], bins[binvar][-1]+hshift),
                    y_range=roi_waste_y_range,
                    y_axis_type='linear'
                )
         
                bincenters = (bins[binvar][:-1]+bins[binvar][1:])/2

                for isuf,suf in enumerate(('_def', '_roi')):
                    if self.mode != 'both': #handle the single chain case
                        if isuf>0:
                            break
                        suf = ''
                        
                    opt = dict(x=bincenters+self._x_shift(isuf,binvar,len(avars)),
                               color=self.palette[isuf])

                    k = np.array(values[binvar][vvv+suf]['sum'])
                    n = np.array(values[binvar][vvv+suf]['size'])
                    nanlen = np.array(values[binvar][vvv+suf]['nanlen'])
     
                    cp_interv = [self._clopper_pearson(kk, nn-ll) for kk,nn,ll in zip(k,n,nanlen)]
                    errlo = [cp[0] for cp in cp_interv]
                    errup = [cp[1] for cp in cp_interv]
                    
                    # errors
                    source = bmd.ColumnDataSource(
                        data=dict(base=bincenters+self._x_shift(isuf,binvar,len(avars)),
                                  errup=errup, errdown=errlo))
                    quant = bmd.Whisker(base='base', upper='errup', lower='errdown', source=source,
                                        level='annotation', line_width=2, line_color=self.palette[isuf])
                    quant.upper_head.size=self.whisker_head_size
                    quant.lower_head.size=self.whisker_head_size
                    quant.upper_head.line_color = self.palette[isuf]
                    quant.lower_head.line_color = self.palette[isuf]
                    feff[figname].add_layout(quant)

                    # central value
                    eff = np.nan_to_num(k/(n-nanlen), nan=0.)
                    if self.mode == 'both':
                        feff[figname].circle(y=eff, size=10, legend_label=leglab[vvv+suf], **opt)
                    else:
                        feff[figname].circle(y=eff, size=10, **opt)
                    feff[figname].line(line_width=1, **opt)
                    feff[figname].line(x=[bincenters[0]-hshift, bincenters[-1]+4*hshift], y=[1.,1.],
                                       line_width=1, line_dash='dashed', color='gray')
              
                feff[figname].xaxis.axis_label = self.info[binvar]['label_2d']
                feff[figname].yaxis.axis_label = ('Seeding Efficiency' if 'has_seeds' in vvv
                                                  else 'Fraction of events w/ #Seeds < #CS regions')
                self._set_fig_common_attributes(
                    feff[figname],
                    title="Seeding Efficiency" if "has_seeds" in vvv else "#Seeds < #CS regions",
                    legend=True if self.mode=='both' else False, location='top_left')

        lay_list = [[fcounts['pt'], feff['pt_has_seeds'], feff['pt_less_seeds']],
                    [fcounts['en'], feff['en_has_seeds'], feff['en_less_seeds']],
                    [fcounts['eta'], feff['eta_has_seeds'], feff['eta_less_seeds']],
                    [fcounts['phi'], feff['phi_has_seeds'], feff['phi_less_seeds']]]
        self._display(lay_list, pars, nevents=df.shape[0], no_cluster=True)
     
        return None

    def resolution_plotter(self, df, pars):
        self._output('res_'+self.tag, pars, folder='Resolution')
        deltaRs = self.cfg['valid_cluster']['tcDeltaRthresh']
        bins_1d, bins_2d = self._get_bins(df)

        ## 1D
        aggr_1d = ['size',] #'std', 'mean',]
        if self.mode == 'both':
            vars_1d = {'clres{}_def': 'Clusters (R/z,'+self.uc['phi']+')', 'clres{}_cs': 'Clusters (CS)',
                       'tcallres{}': 'TCs'
                       }
        else:
            vars_1d = {'clres{}': 'Clusters', 'tcallres{}': 'TCs'}

        centers_1d, values_1d = ({} for _ in range(2))
        for x in bins_1d.keys():
            centers_1d.update({x: self.bincenters(bins_1d[x])})
            for v in vars_1d.keys():
                cuts_1d = pd.cut(df[v.format(x)], bins_1d[x], labels=centers_1d[x], include_lowest=True)
                values_1d.update({v.format(x): df.groupby(cuts_1d).agg(aggr_1d)[v.format(x)]})

        ## 2D
        aggr_2d = ['median', 'mean', self._q1, self._q3, 'std', common.std_eff, 'size']
        if self.mode == 'both':
            vars_2d = ['clres{}_'+x for x in ('def', 'cs')]
        else:
            vars_2d = ['clres{}']
        vars_2d += ['tcallres{}'] + ['tc' + str(t).replace('.','p') + 'res{}' for t in deltaRs]
        
        centers_2d, cuts_2d, values_2d = ({} for _ in range(3))
        for x in bins_2d.keys():
            centers_2d.update({x: self.bincenters(bins_2d[x], self.info[x]['round']).astype(str)})
            cuts_2d.update({x: pd.cut(df['gen'+x], bins_2d[x],
                                      labels=centers_2d[x], include_lowest=True)})
            values_2d.update({x: df.groupby(cuts_2d[x])
                              .agg(aggr_2d)
                              .rename(columns={'<lambda_0>': 'q1',
                                               '<lambda_1>': 'q3'})[[k.format('en') for k in vars_2d]]})
        
        common_y_range = {'median':  bmd.Range1d(-0.3, 0.05),
                          'mean':    bmd.Range1d(-0.3, 0.05),
                          'std':     bmd.Range1d(0., 0.12),
                          'std_eff': bmd.Range1d(0., 0.12)}
        violins = {}
        for bk in bins_2d.keys():
            ibins = np.tile(cuts_2d[bk].to_numpy(), len(vars_2d)).astype(float)
            if self.mode == 'both':
                concat_list = [np.array(['Clusters (R/z,'+self.uc['phi']+')' for _ in range(len(df['clresen_def']))]),
                               np.array(['Clusters (CS)' for _ in range(len(df['clresen_cs']))]),
                               np.array(['All TCs'  for _ in range(len(df['tcallresen']))])]
            else:
                concat_list = [np.array(['Clusters' for _ in range(len(df['clresen']))]),
                               np.array(['All TCs'  for _ in range(len(df['tcallresen']))])]

            for it,t in enumerate(deltaRs):
                concat_list.append(np.array(['TCs (dR<{})'.format(t) for _ in range(len(df[vars_2d[it+2].format('en')]))]))
            group_ids = np.concatenate(concat_list, axis=None)

            if self.mode == 'both':
                vals_list = [df['clresen_def'], df['clresen_cs']]
            else:
                vals_list = [df['clresen']]
            vals_list += [df['tcallresen']]
            for it,t in enumerate(deltaRs):
                vals_list.append(df['tc' + str(t).replace('.','p') + 'resen'])
            group_vals = pd.concat(vals_list)
            
            violins[bk] = hv.Violin((ibins.tolist(), group_ids, group_vals.to_numpy()), [bk, 'Group'])

            violins[bk] = violins[bk].opts(
                opts.Violin(height=self.fig_height, width=1200,
                            violin_color=hv.dim('Group').str(), #Category10 by default, can't change it
                            show_legend=True,
                            box_color='black',
                            cut=0.4, # proxy for when to stop calculating the distribution tails
                            tools=[self.fig_tools], default_tools=[],
                            ), 
                clone=True,)
            violins[bk] = hv.render(violins[bk]) # convert to bokeh's format
            violins[bk].y_range = common_y_range['median']
     
        figs_1d = {x:None for x in bins_2d.keys()}
        quants_2d = ('median', 'mean', 'std', 'std_eff')
        figs_2d = {x: {z:None for z in quants_2d} for x in bins_2d.keys()}
        if self.mode == 'both':
            leglab = {'clres_def': 'Clusters (R/z,'+self.uc['phi']+')', 'clres_cs': 'Clusters (CS)'}
        else:
            leglab = {'clres': 'Clusters'}
        leglab.update({'tcallres': 'All TCs'})
        for t in deltaRs:
            leglab.update({'tc'+str(t).replace('.','p')+'res': 'TCs (dR<{})'.format(t)})
        assert len(leglab.keys()) == len(vars_2d)
                    
        for bk in bins_2d.keys():
            hshift = (bins_2d[bk][1]-bins_2d[bk][0])/2
            ymax = max([np.array(values_2d[bk][x.format('en')]['q3']).max() for x in vars_2d])
            ymin = min([np.array(values_2d[bk][x.format('en')]['q1']).min() for x in vars_2d])
            figs_opt = dict(width=700, height=self.fig_height, title='',
                            tools=self.fig_tools, y_axis_type='linear',)

            figs_1d[bk] = bokeh.plotting.figure(
                x_range=(bins_1d[bk][0], bins_1d[bk][-1]),
                **figs_opt
            )
            figs_1d[bk].xaxis.axis_label = self.info[bk]['label_1d']
            figs_1d[bk].yaxis.axis_label = 'Counts'

            for qt in quants_2d:
                figs_2d[bk][qt] = bokeh.plotting.figure(
                    x_range=(bins_2d[bk][0], bins_2d[bk][-1]+hshift),
                    y_range=common_y_range[qt],
                    **figs_opt
                )

            ymaxline = 0
            for iv1d, (v1d, lab) in enumerate(vars_1d.items()):
                size = list(values_1d[v1d.format(bk)]['size'])
                #mean = list(values_1d[v1d.format(bk)]['mean'])
                #std  = list(values_1d[v1d.format(bk)]['std'])

                hshift = self._x_shift(iv1d, bk+'res', len(vars_1d))
                # errors
                source = bmd.ColumnDataSource(
                    data=dict(base=centers_1d[bk]+hshift,
                              errup=size+np.sqrt(size)/2,
                              errdown=size-np.sqrt(size)/2))
                quant = bmd.Whisker(base='base', upper='errup', lower='errdown', source=source,
                                    level='annotation', line_width=2, line_color=self.palette[iv1d])
                quant.upper_head.size = self.whisker_head_size
                quant.lower_head.size = self.whisker_head_size
                quant.upper_head.line_color = self.palette[iv1d]
                quant.lower_head.line_color = self.palette[iv1d]
                figs_1d[bk].add_layout(quant)

                opt = dict(x=centers_1d[bk]+hshift, y=size, legend_label=lab, color=self.palette[iv1d])
                figs_1d[bk].circle(size=10, **opt)
                figs_1d[bk].line(line_width=1, **opt)
                ymaxline = max(ymaxline, max(size) + (max(size)-min(size))/15)

            yrange = (-ymaxline/20, ymaxline)
            figs_1d[bk].line(x=[0., 0.], y=yrange, line_width=1, line_dash='dashed', color='gray')
            figs_1d[bk].y_range = bmd.Range1d(*yrange)
            self._set_fig_common_attributes(figs_1d[bk], title=self.info[bk]['title_1d'], location='top_left')
            
            for ivar, avar in enumerate(vars_2d):
                bincenters = (bins_2d[bk][:-1]+bins_2d[bk][1:])/2
                opt = dict(x=bincenters+self._x_shift(ivar,bk,len(vars_2d)),
                           legend_label=leglab[avar.format('')], color=self.palette[ivar])

                xmax = len(vars_2d)*self.info[bk]['nbins_2d']+1.5*self.info[bk]['nbins_2d']
                violins[bk].line(x=[0., xmax], y=[0.,0.],
                                 line_width=1, line_dash='dashed', color='gray')
                violins[bk].xaxis.major_label_text_alpha = 0.
                violins[bk].xaxis.major_label_text_font_size = '0pt'
                violins[bk].xaxis.major_label_text_color = 'white'
                violins[bk].xaxis.major_tick_line_alpha = 0.
                violins[bk].xaxis.major_tick_line_color = 'white'
                violins[bk].xaxis.axis_label = self.info[bk]['label_2d']
                violins[bk].yaxis.axis_label = r"$$E/E_{Gen}-1" + r"\:\:\text{" + "(median)" + r"}" + "$$"
                
                for qt in quants_2d:
                    qt_arr = np.array(values_2d[bk][avar.format('en')][qt])                
                    qt_opt = dict(y=qt_arr)
     
                    figs_2d[bk][qt].circle(size=10, **qt_opt, **opt)
                    figs_2d[bk][qt].line(line_width=1, **qt_opt, **opt)
                    figs_2d[bk][qt].xaxis.axis_label = self.info[bk]['label_2d']
                    
                    if qt != 'median' and qt != 'mean':
                        if qt == 'std':
                            figs_2d[bk][qt].yaxis.axis_label = 'Standard deviation'
                        elif qt == 'std_eff':
                            figs_2d[bk][qt].yaxis.axis_label = 'Effective standard deviation'
                        continue

                    figs_2d[bk][qt].yaxis.axis_label = r"$$E/E_{Gen}-1" + r"\:\:\text{(" + qt + r")}" + "$$"
                    figs_2d[bk][qt].line(x=[bincenters[0]-hshift, bincenters[-1]+10*hshift], y=[0.,0.],
                                         line_width=1, line_dash='dashed', color='gray')

                    # quantiles                                    
                    q1 = np.array(values_2d[bk][avar.format('en')]['q1'])
                    q3 = np.array(values_2d[bk][avar.format('en')]['q3'])

                    source = bmd.ColumnDataSource(
                        data=dict(base=bincenters+self._x_shift(ivar,bk,len(vars_2d)),
                                  q3=q3, q1=q1))
                    quant = bmd.Whisker(base='base', upper='q3', lower='q1', source=source,
                                        level='annotation', line_width=2, line_color=self.palette[ivar])
                    quant.upper_head.size=10
                    quant.lower_head.size=10
                    quant.upper_head.line_color = self.palette[ivar]
                    quant.lower_head.line_color = self.palette[ivar]
                    figs_2d[bk][qt].add_layout(quant)

        for bk in bins_2d.keys():
            self._set_fig_common_attributes(violins[bk], title=self.info[bk]['title_2d'], show_grid=False)
            for qt in quants_2d:
                if qt in ('std', 'std_eff'):
                    loc = "top_left"
                else:
                    loc = "bottom_right"
                self._set_fig_common_attributes(figs_2d[bk][qt], title=self.info[bk]['title_2d'], location=loc)

        # create display layout of all the figures
        lay_list = [[figs_1d[v] for v in ('en', 'eta', 'phi', 'pt')]]
        for v in ('en', 'eta', 'phi', 'pt'):
            row = [figs_2d[v][qt] for qt in quants_2d] + [violins[v]]
            lay_list.append(row)
        self._display(lay_list, pars, nevents=df.shape[0])
     
        return None

    def distribution_plotter(df, pars, user):
        self._output('dist_'+self.tag, pars)
        allfigs = []
        varpairs = (('cl', 'tc0p1'),  ('clres', 'tc0p1res'))

        bins, _ = self._get_bins(self, df)
        for bv in self.info.keys():
            bins[bv] = {'def': bins[bv],
                        'res': np.linspace(-0.6, 0, self.info[bv]['nbins_1d']+1)}

        wshifts = {'eta': {'cl': 0.01, 'tc0p1': -0.01, 'clres': 0.002, 'tc0p1res': -0.002},
                   'phi': {'cl': 0.01, 'tc0p1': -0.1,  'clres': 0.002, 'tc0p1res': -0.002, },
                   'en':  {'cl': 10,   'tc0p1': -10,   'clres': 0.002, 'tc0p1res': -0.002},
                   'pt':  {'cl': 0,   'tc0p1': 0,   'clres': 0.0, 'tc0p1res': -0.0}}

        aggr_vars = ['mean', 'std', 'size']
        for avars in varpairs:
            vstr = 'def' if avars==('cl', 'tc') else 'res'
            values = {x: {avars[0]: df.groupby(pd.cut(df[avars[0]+x], bins[x][vstr])).agg(aggr_vars),
                          avars[1]: df.groupby(pd.cut(df[avars[1]+x], bins[x][vstr])).agg(aggr_vars)}
                      for x in bins.keys()}
     
            figs = {x:None for x in labels.keys()}
            values_d = {avars[0]: ('Clusters', self.palette[0]),
                        avars[1]: ('TCs', self.palette[1])}
     
            for il in labels.keys():
                hshift = (bins[il][vstr][1]-bins[il][vstr][0])/2
                figs[il] = bokeh.plotting.figure(
                    width=600, height=self.fig_height, title='', tools=self.fig_tools,
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
                                                            errup=size+np.sqrt(size)/2,
                                                            errdown=size-np.sqrt(size)/2))
                    quant = bmd.Whisker(base='base', upper='errup', lower='errdown', source=source,
                                        level='annotation', line_width=2, line_color=values_d[avar][1])
                    quant.upper_head.size=10
                    quant.lower_head.size=10
                    quant.upper_head.line_color = values_d[avar][1]
                    quant.lower_head.line_color = values_d[avar][1]
                    figs[il].add_layout(quant)
     
                    figs[il].circle(size=10, **opt)
                    figs[il].line(line_width=1, **opt)
                    
                self._set_fig_common_attributes(figs[il], title=self.info[bk]['title_1d'])
     
            allfigs.append(figs)

        lay_list = [[allfigs[0]['en'],  allfigs[1]['en']],
                    [allfigs[0]['phi'], allfigs[1]['phi']],
                    [allfigs[0]['eta'], allfigs[1]['eta']],
                    [allfigs[0]['pt'],  allfigs[1]['pt']]]
        self._display(lay_list, pars, nevents=df.shape[0])
     
        return None

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

    bokeh.io.output_file(plot_name)
    
    stats_fig  = stats_plotter(pars=FLAGS.params, names_d=names_d)
    res_figs, res_ratios, slider = resolution_plotter(pars=FLAGS.params, names_d=names_d)

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
