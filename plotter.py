"""
Some plots.
"""
import os
import numpy as np
from bokeh.layouts import column
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, show
class Plotter:
    def __init__(self, outname='plots/plotter_out.hdf5', **kw):
        self.outname = outname
        self.kw = kw
        self.bincounts = []
        self.histos = []
        self.orig_data = None

        self.remove_output()
        
    def remove_output(self):
        if os.path.exists(self.outname):
            os.system('rm {}'.format(self.outname))
        self.bincounts = []
        self.histos = []

    def save_orig_data(self, data):
        """
        Plots the original number of trigger cells per bin.
        `data` should be a list of trigger cell phi positions, one per R/z slice.
        """
        if isinstance(data, (tuple,list)):
            raise ValueError('This method cannot be applied to a data list.')
        bins = np.digitize( data, bins=self.kw['PhiBinEdges'])
        bincounts = np.bincount(bins, minlength=self.kw['NbinsPhi'])
        self.orig_data = bincounts

    def save_gen_data(self, data):
        """
        Plots the number of trigger cells per bin as output by the learning framework.
        `data` should be a list of trigger cell phi positions, one per R/z slice.
        """
        if not isinstance(data, (tuple,list)):
            data = [ data ]

        for i,rzslice in enumerate(data):
            bins = np.digitize( rzslice, bins=self.kw['PhiBinEdges'])
            bincounts = np.bincount(bins, minlength=self.kw['NbinsPhi'])

            self.bincounts.append( bincounts )
            self.histos.append( np.histogram(bincounts, bins=20) )

    def plot(self, plot_name='plots/ntriggercells.html', show_html=True):
        """Plots the data collected by the save_* methods."""
        assert self.orig_data is not None
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
        s_bincounts = { 'curr_x': np.arange(nelems_max),
                        'curr_y': binspad/sum(binspad) + logpad } # normalize to unity
        s_histocounts = {}
        s_histoedges = {}

        for i,(bc,histo) in enumerate(zip(self.bincounts,self.histos)):
            bc_padded = np.pad(bc, pad_width=(0,nelems_max-len(bc)))
            s_bincounts.update({'bc'+str(i): (bc_padded)/sum(bc_padded) + logpad}) # normalize to unity
            s_histocounts.update({'hcounts'+str(i): histo[0]})
            s_histoedges.update({'hedges'+str(i): histo[1]})

        s_bincounts   = ColumnDataSource(data=s_bincounts)
        # s_histocounts = ColumnDataSource(data=s_histocounts)
        # s_histoedges  = ColumnDataSource(data=s_histoedges)
        plot_gen = figure(width=1200, height=300, y_range=(logpad,1), y_axis_type="log")
        plot_gen.circle('curr_x', 'curr_y', source=s_bincounts)
        
        callback = CustomJS(args=dict(source=s_bincounts),
                            code="""
                            const data = source.data;
                            const slval = cb_obj.value;
                            const numberstr = slval.toString();
                            data['curr_y'] = data['bc'+numberstr];
                            source.change.emit();
                            """)

        slider = Slider(start=0, end=self.kw['Epochs'],
                        value=0, step=1., title='Epoch')
        slider.js_on_change('value', callback)
        
        # for i,(bc,histo) in enumerate(zip(self.bincounts,self.histos)):
        #     b.graph(idx=i, data=[np.arange(len(bc)),bc],
        #             style='circle', color='orange', line=False)
        #     #b.histogram( idx=i, data=histo, color='orange' )

        assert sum(self.orig_data) == sum(self.bincounts[0])
        plot_orig = figure(width=1200, height=300, y_range=(logpad,1), y_axis_type="log")
        plot_orig.circle(np.arange(len(self.orig_data)), self.orig_data/sum(self.orig_data),
                         color='black')

        layout = column( slider, plot_gen, plot_orig )
        if show_html:
            show(layout)
            

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
