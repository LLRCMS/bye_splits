"""
Some plots.
"""
import os
import numpy as np
import pandas as pd
from bokeh.layouts import column
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, show, save
class Plotter:
    def __init__(self, outname='plots/plotter_out.hdf5', **kw):
        self.outname = outname
        self.kw = kw
        self.bincounts = []
        self.histos = []
        self.gen_data = []
        self.orig_data = None
        self.orig_data_counts = None

        self.remove_output()

        self.plot_min = 1e10
        self.plot_max = 0
        
    def remove_output(self):
        if os.path.exists(self.outname):
            os.system('rm {}'.format(self.outname))
        self.bincounts = []
        self.histos = []

    def save_orig_data(self, data, bins=None, minlength=None):
        """
        Plots the original number of trigger cells per bin.
        `data` should be a list of trigger cell phi positions, one per R/z slice.
        """
        if isinstance(data, (tuple,list)):
            raise ValueError('This method cannot be applied to a data list.')

        self.orig_data = data
        bins = self.kw['PhiBinEdges'] if bins is None else bins
        minlength = self.kw['NbinsPhi'] if minlength is None else minlength
        
        digi_bins = np.digitize(data, bins=bins)
        bincounts = np.bincount(digi_bins, minlength=minlength)
        assert bincounts[0] == 0 #no out-of-bound values are possible
        bincounts = bincounts[1:]
        
        # print(data)
        # print(min(data), max(data))
        # print(bins[0], bins[-1], len(bins))
        # print(self.kw['NbinsPhi'])
        # print()
        # print(min(digi_bins), max(digi_bins), np.unique(digi_bins), len(np.unique(digi_bins)))
        # print(bincounts, len(bincounts))
        print(sum(bincounts))
        print(len(data))
        quit()



        assert len(bincounts) == self.kw['NbinsPhi']

        self.orig_data_counts = bincounts

    def save_gen_data(self, data, bins=None, minlength=None):
        """
        Plots the number of trigger cells per bin as output by the learning framework.
        `data` should be a list of trigger cell phi positions, one per R/z slice.
        """
        self.gen_data.append(data)
        #find the overall min and max of the histogram to be plotted
        bins = self.kw['PhiBinEdges'] if bins is None else bins
        minlength = self.kw['NbinsPhi'] if minlength is None else minlength
        digi_bins = np.digitize(data, bins=bins)
        bincounts = np.bincount(digi_bins, minlength=minlength)
        bincounts = bincounts[1:] #do not plot out-of-bound values
        if self.plot_max < np.max(bincounts):
            self.plot_max = np.max(bincounts)
        if self.plot_min > np.min(bincounts):
            self.plot_min = np.min(bincounts)

        if not isinstance(data, (tuple,list)):
            # m = ( '[plotter.py] The bins should cover the {s} data value!' +
            #       ' However: {q1} {q2} {q3}.' )
            # if max(data) >= bins[-1]:
            #     raise ValueError(m.format(s='maximum', q1=max(data), q2='>=', q3=bins[-1]))
            # if min(data) < bins[0]:
            #     raise ValueError(m.format(s='minimum', q1=min(data), q2='<', q3=bins[0]))
            data = [ data ]

        minlength = self.kw['PhiBinEdges'] if minlength is None else minlength
        
        for i,rzslice in enumerate(data):
            #print(min(rzslice), max(rzslice), bins[-1])
            digi_bins = np.digitize(rzslice, bins=bins)
            #print(max(digi_bins), min(digi_bins), len(bins), len(rzslice))
            #print(digi_bins, len(digi_bins))
            bincounts = np.bincount(digi_bins, minlength=minlength)
            #print(bincounts)

            self.bincounts.append( bincounts )
            self.histos.append( np.histogram(bincounts, bins=20) )

    def plot(self, plot_name='plots/ntriggercells.html',
             minval=None, maxval=None,
             density=False, show_html=False):
        """Plots the data collected by the save_* methods."""
        assert self.orig_data_counts is not None
        assert len(self.bincounts) > 0
        logpad = 1e-4

        # if the out bin number is larger than `NbinsPhi` (makes no sense but can happen)
        # the shapes will not match
        nelems_max = 0
        for bc in self.bincounts:
            if len(bc) > nelems_max:
                nelems_max = len(bc)

        binspad = np.pad(self.bincounts[0],
                         pad_width=(0,nelems_max-len(self.bincounts[0])))
        s_bincounts = { 'curr_x': np.arange(nelems_max) }
        if density:
            s_bincounts.update( {'curr_y': binspad/sum(binspad) + logpad} )
        else:
            s_bincounts.update( {'curr_y': binspad} )

        nbins_plot = 150
        s_diff = {}
        hist, edges = np.histogram(self.orig_data-self.gen_data[0],
                                   density=False, bins=nbins_plot)
        df_diff = pd.DataFrame( {'diff_default': hist,
                                 'leftedge_default': edges[1:],
                                 'rightedge_default': edges[:-1]} )
        for i,elem in enumerate(self.gen_data):
            hist, edges = np.histogram(self.orig_data-elem,
                                       density=False, bins=nbins_plot)

            df_diff['diff'+str(i)] = hist
            df_diff['leftedge'+str(i)] = edges[1:]
            df_diff['rightedge'+str(i)] = edges[:-1]
        s_diff = ColumnDataSource(df_diff)
        
        for i,bc in enumerate(self.bincounts):
            bc_padded = np.pad(bc, pad_width=(0,nelems_max-len(bc)))
            if density:
                s_bincounts.update({'bc'+str(i): (bc_padded)/sum(bc_padded) + logpad})
            else:
                s_bincounts.update({'bc'+str(i): (bc_padded)})

        s_bincounts = ColumnDataSource(data=s_bincounts)
        y_axis_type = 'log' if density else 'linear'
        plot_opt = dict(width=1200, height=300)
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

        plot_distr.circle('curr_x', 'curr_y', source=s_bincounts,
                          legend_label='Output')

        callback = CustomJS(args=dict(s1=s_bincounts, s2=s_diff),
                            code="""
                            const d1 = s1.data;
                            const d2 = s2.data;
                            const slval = cb_obj.value;
                            const numberstr = slval.toString();
                            d1['curr_y'] = d1['bc'+numberstr];
                            d2['diff_default'] = d2['diff'+numberstr];
                            d2['leftedge_default'] = d2['leftedge'+numberstr];
                            d2['rightedge_default'] = d2['rightedge'+numberstr];
                            s1.change.emit();
                            s2.change.emit();
                            """)

        assert sum(self.orig_data_counts) == sum(self.bincounts[0])

        plot_distr.circle(np.arange(len(self.orig_data_counts)),
                          #self.orig_data_counts/sum(self.orig_data_counts),
                          self.orig_data_counts,
                          color='black',
                          legend_label='Input')

        plot_diff = figure(**plot_opt)
        plot_diff.quad( top='diff_default', bottom=0,
                        left='leftedge_default', right='rightedge_default',
                        fill_color='navy', line_color='white', alpha=0.5,
                        source=s_diff )

        slider = Slider(start=0, end=self.kw['Epochs'],
                        value=0, step=1., title='Epoch')
        slider.js_on_change('value', callback)

        layout = column( slider, plot_distr, plot_diff )
        if show_html:
            show(layout)
        else:
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
