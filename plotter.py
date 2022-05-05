"""
Some plots.
"""
import os
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, show, save
from bokeh.models import Panel, Tabs, Label
from bokeh.io import output_file

class Plotter:
    def __init__(self, outname='plots/plotter_out.hdf5', **kw):
        self.outname = outname
        self.kw = kw

        self.plot_min = 1e10
        self.plot_max = 0
        self.dim_bins = (1600, 300)
        self.dim_phis = (1200, 300)
        self.margin = (10,70)

        self.phi_tabs, self.bin_tabs = ([] for _ in range(2))

        self.reset()
    
    def reset(self):
        """Prepare for a new tab."""
        self.histos = []
        self.gen_data = []
        self.gen_data_counts = []
        self.gen_bins_counts = []
        self.gen_bins_counts_sum = []
        self.orig_data = None
        self.orig_data_counts = None
        self.orig_bins = None
        self.phi_old = None
        self.phi_new = []
        
    def save_orig_data(self, data, data_type, boundary_sizes, bins=None, minlength=None):
        """
        Plots the original number of trigger cells per bin.
        `data` should be a list of trigger cell phi positions, one per R/z slice.
        """
        assert data_type=='data' or data_type=='bins'
        
        if isinstance(data, (tuple,list)):
            raise ValueError('This method cannot be applied to a data list.')

        data = self._remove_duplicated_boundaries(data, boundary_sizes)

        if data_type == 'data':
            self.orig_data = data
            bins = self.kw['PhiBinEdges'] if bins is None else bins
            minlength = self.kw['NbinsPhi'] if minlength is None else minlength

            digi_bins = np.digitize(data, bins=bins)
            bincounts = np.bincount(digi_bins, minlength=minlength)
            assert bincounts[0] == 0 #no out-of-bound values are possible
            bincounts = bincounts[1:]

            assert len(bincounts) == self.kw['NbinsPhi']

            self.orig_data_counts = bincounts

        elif data_type == 'bins':
            self.orig_bins = data

    def _remove_duplicated_boundaries(self, data, bound_size):
        return data[bound_size:]
        
    def save_gen_data(self, data, boundary_sizes, data_type, bins=None, minlength=None):
        """
        Plots the number of trigger cells per bin as output by the learning framework.
        `data` should be a list of trigger cell phi positions, one per R/z slice.
        """
        assert data_type=='data' or data_type=='bins'
        
        if isinstance(data, (tuple,list)):
            raise ValueError('Ups.')

        data = self._remove_duplicated_boundaries(data, boundary_sizes)

        if data_type == 'data':
            self.gen_data.append(data)
        elif data_type == 'bins':
            self.gen_bins_counts.append(data)
            self.gen_bins_counts_sum.append( np.sum(data) )

        if data_type == 'data':
            #find the overall min and max of the histogram to be plotted
            bins = self.kw['PhiBinEdges'] if bins is None else bins

            #consider the two out-of-bound bins
            minlength = self.kw['NbinsPhi']+2 if minlength is None else minlength

            _digi_bins = np.digitize(data, bins=bins)        
            _bincounts = np.bincount(_digi_bins, minlength=minlength)
            _bincounts = _bincounts[1:-1] #do not plot out-of-bound values
        elif data_type == 'bins':
            _bincounts = data

        if self.plot_max < np.max(_bincounts):
            self.plot_max = np.max(_bincounts)
        if self.plot_min > np.min(_bincounts):
            self.plot_min = np.min(_bincounts)

        if data_type == 'data':
            digi_bins = np.digitize(data, bins=bins)
            bincounts = np.bincount(digi_bins, minlength=minlength)
            assert len(bincounts)==minlength
            assert sum(self.orig_data_counts) >= sum(bincounts)

            #remove the out-of-bounds from the plot
            bincounts = bincounts[1:-1]
            self.gen_data_counts.append( bincounts )
            self.histos.append( np.histogram(bincounts, bins=20) )

    def plot_nn(self, plot_name='plots/ntriggercells.html',
                minval=None, maxval=None,
                density=False, show_html=False):
        """Plots the data collected by the save_* methods."""
        assert self.orig_data is not None
        assert self.orig_bins is not None
        assert self.orig_data_counts is not None
        assert len(self.gen_data_counts) > 0
        assert len(self.gen_bins_counts) > 0
        assert len(self.gen_bins_counts_sum) > 0
        logpad = 1e-4

        # if the out bin number is larger than `NbinsPhi` (makes no sense but can happen)
        # the shapes will not match
        nelems_max = 0
        for bc in self.gen_data_counts:
            if len(bc) > nelems_max:
                nelems_max = len(bc)

        binspad = np.pad(self.gen_data_counts[0],
                         pad_width=(0,nelems_max-len(self.gen_data_counts[0])))

        # data counts start display
        s_data_counts = { 'curr_x': np.arange(nelems_max) }
        if density:
            s_data_counts.update( {'curr_y': binspad/sum(binspad) + logpad} )
        else:
            s_data_counts.update( {'curr_y': binspad} )

        # bin counts start display
        gen0 = self.gen_bins_counts[0]
        s_bins_counts = { 'curr_x': np.arange(len(gen0)) }
        if density:
            s_bins_counts.update( {'curr_y': gen0/sum(gen0) + logpad} )
        else:
            s_bins_counts.update( {'curr_y': gen0} )

        nbins_plot = 150
        s_diff = {}
        hist, edges = np.histogram(self.orig_data-self.gen_data[0],
                                   density=False, bins=nbins_plot)
        df_diff = pd.DataFrame( {'diff_default': hist,
                                 'leftedge_default': edges[1:],
                                 'rightedge_default': edges[:-1]} )

        hists, lefts, rights = ([] for _ in range(3))
        for i,elem in enumerate(self.gen_data):
            hist, edges = np.histogram(self.orig_data-elem,
                                       density=False, bins=nbins_plot)
            hists.append( pd.DataFrame(hist, columns=['diff'+str(i)]) )
            lefts.append( pd.DataFrame(edges[1:], columns=['leftedge'+str(i)]) )
            rights.append( pd.DataFrame(edges[:-1], columns=['rightedge'+str(i)]) )

        hists_to_concat = [df_diff]
        hists_to_concat.extend(hists)
        hists_to_concat.extend(lefts)
        hists_to_concat.extend(rights)
        df_diff = pd.concat(hists_to_concat, axis=1)
        s_diff = ColumnDataSource(df_diff)
        
        for i,bc in enumerate(self.gen_data_counts):
            bc_padded = np.pad(bc, pad_width=(0,nelems_max-len(bc)))
            if density:
                s_data_counts.update({'bc'+str(i): (bc_padded)/sum(bc_padded) + logpad})
            else:
                s_data_counts.update({'bc'+str(i): (bc_padded)})

        for i,bc in enumerate(self.gen_bins_counts):
            if density:
                s_bins_counts.update({'bc'+str(i): (bc)/sum(bc) + logpad})
            else:
                s_bins_counts.update({'bc'+str(i): (bc)})

        s_data_counts = ColumnDataSource(data=s_data_counts)
        s_bins_counts = ColumnDataSource(data=s_bins_counts)
        
        y_axis_type = 'log' if density else 'linear'
        plot_opt = dict(width=self.dim_bins[0], height=self.dim_bins[1])
        if minval is None or maxval is None:
            if density: 
                plot_distr = figure( y_range=(logpad,1), y_axis_type=y_axis_type,
                                     **plot_opt)
            else:
                plot_distr = figure( y_axis_type=y_axis_type, **plot_opt)
                #(self.plot_min,self.plot_max)
        else:
            y_range = (logpad,1) if density else (minval,maxval)
            plot_distr = figure( y_range=y_range, y_axis_type=y_axis_type,
                               **plot_opt)

        plot_distr.circle('curr_x', 'curr_y', source=s_data_counts,
                          legend_label='Data Output', color='red')
        plot_distr.triangle('curr_x', 'curr_y', source=s_bins_counts,
                            legend_label='Bins Output', color='red')
        plot_distr.legend.click_policy='hide'

        callback = CustomJS(args=dict(s1=s_data_counts, s2=s_diff, s3=s_bins_counts),
                            code="""
                            const d1 = s1.data;
                            const d2 = s2.data;
                            const d3 = s3.data;
                            const slval = cb_obj.value;
                            const numberstr = slval.toString();
                            d1['curr_y'] = d1['bc'+numberstr];
                            d3['curr_y'] = d3['bc'+numberstr];
                            d2['diff_default'] = d2['diff'+numberstr];
                            d2['leftedge_default'] = d2['leftedge'+numberstr];
                            d2['rightedge_default'] = d2['rightedge'+numberstr];
                            s1.change.emit();
                            s2.change.emit();
                            s3.change.emit();
                            """)

        # some generated data can lie outside the bin leftmost and rightmost edges
        assert sum(self.orig_data_counts) >= sum(self.gen_data_counts[0]) 

        plot_distr.circle(np.arange(len(self.orig_data_counts)),
                          #self.orig_data_counts/sum(self.orig_data_counts),
                          self.orig_data_counts,
                          color='black',
                          legend_label='Data Input')

        plot_distr.triangle(np.arange(len(self.orig_bins)),
                            self.orig_bins,
                            color='grey',
                            legend_label='Bins Input')

        plot_diff = figure(**plot_opt)
        plot_diff.quad( top='diff_default', bottom=0,
                        left='leftedge_default', right='rightedge_default',
                        fill_color='navy', line_color='white', alpha=0.5,
                        source=s_diff )

        plot_count = figure(**plot_opt)
        plot_count.line(np.arange(len(self.gen_bins_counts_sum)),
                        self.gen_bins_counts_sum, 
                        legend_label='Bin Output Count Sum',
                        color='black')

        slider = Slider(start=0, end=len(self.gen_data),
                        value=0, step=1., title='Epoch')
        slider.js_on_change('value', callback)

        layout = column( slider, plot_distr, plot_diff, plot_count )
        if show_html:
            show(layout)
        else:
            output_file(plot_name)
            save(layout)

    def save_orig_phi_data(self, phi_old):
        self.phi_old = phi_old

    def save_gen_phi_data(self, phi_new):
        self.phi_new.append(phi_new)

    def save_iterative_phi_tab(self, nonzero_ratio, ncellstot):
        # phi values display
        gen0 = self.phi_old
        phi_s = { 'curr_x': gen0 }
        phi_s.update( {'curr_y': self.phi_new[0]} )
        phi_s.update( {'color': ['gray' if x==0 else 'blue' for x in self.phi_new[0]]} )

        for i, phis in enumerate(self.phi_new):
            phi_s.update({'phi'+str(i): phis})

        phi_s = ColumnDataSource(data=phi_s)

        # plotting phis
        plot_opt = dict(width=self.dim_phis[0], height=self.dim_phis[1])
        p = figure(**plot_opt,
                   tools="hover,pan,box_zoom,reset,save")
        p.circle('curr_x', 'curr_y', color='color', source=phi_s)
        p.xaxis.axis_label = 'Trigger cell index'
        p.yaxis.axis_label = 'Distance travelled in phi'

        self.phi_tabs.append( p )

        text1 = 'Ratio: {0:.2f}'.format(nonzero_ratio)
        text2 = 'NCells: {0}'.format(ncellstot)
        glyph1 = Label(x=self.margin[0], y=self.dim_phis[1]-self.margin[1],
                       text=text1, angle=0.0, text_color='black', text_font_size='9pt',
                       x_units='screen', y_units='screen')
        glyph2 = Label(x=self.margin[0], y=self.dim_phis[1]-self.margin[1]-15,
                       text=text2, angle=0.0, text_color='black', text_font_size='9pt',
                       x_units='screen', y_units='screen')
        p.add_layout(glyph1)
        p.add_layout(glyph2)
        
        p.hover.tooltips = [
            ('id', '@curr_x'),
        ]
        p.toolbar.logo = None
        
    def save_iterative_bin_tab(self, show_html=False):
        """Plots the data collected by the save_* methods."""
        # bin counts start display
        gen0 = self.gen_bins_counts[0]
        bins_s = { 'curr_x': np.arange(len(gen0)) }
        bins_s.update( {'curr_y': gen0} )

        for i,bc in enumerate(self.gen_bins_counts):
            bins_s.update({'bc'+str(i): (bc)})

        bins_s = ColumnDataSource(data=bins_s)

        # plotting bins
        plot_opt = dict(width=1200, height=300)
        p = figure( **plot_opt)

        p.triangle(np.arange(len(self.orig_bins)), self.orig_bins,
                   color='black', legend_label='Bins Input')
        p.triangle('curr_x', 'curr_y', source=bins_s,
                   legend_label='Bins Output', color='red')
        p.legend.click_policy='hide'
        p.xaxis.axis_label = r'$$\text{Bin index (}[-\pi;\pi[\text{)}$$'
        p.yaxis.axis_label = 'Counts'

        self.bin_tabs.append( p )
        
        # callback = CustomJS(args=dict(s3=bins_s),
        #                     code="""
        #                     const d3 = s3.data;
        #                     const slval = cb_obj.value;
        #                     const numberstr = slval.toString();
        #                     d3['curr_y'] = d3['bc'+numberstr];
        #                     s3.change.emit();
        #                     """)

        # slider = Slider(start=0, end=len(self.gen_bins_counts),
        #                 value=0, step=1., title='Epoch')
        # slider.js_on_change('value', callback)

        p.toolbar.logo = None

    def plot_iterative(self, plot_name, tab_names, show_html):
        assert len(self.bin_tabs)==len(self.phi_tabs)
        self.tabs = []
        for name, pt, bt in zip(tab_names, self.phi_tabs, self.bin_tabs):
            l = column(pt, bt)
            tb = Panel(child=l, title=name)
            self.tabs.append( tb )
        
        layout = Tabs(tabs=self.tabs)
        if show_html:
            show(layout)
        else:
            output_file(plot_name)
            save(layout)

if __name__ == "__main__":
    import h5py
    from airflow.airflow_dag import optimization_kwargs
    from data_processing import preprocess
    
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')

    assert len(store_in.keys()) == 1
    data, _, _ = preprocess(
        data=store_in['data'],
        nbins_phi=kw['NbinsPhi'],
        nbins_rz=kw['NbinsRz'],
        window_size=kw['WindowSize']
    )
    rzslices = [ rzslice[:,0] for rzslice in data ]
    assert len(optimization_kwargs['FesAlgos'])==1
    plotter = Plotter()
    plotter.save( rzslices, save=False, **optimization_kwargs )
    plotter.plot()
