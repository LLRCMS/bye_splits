"""
Some plots.
"""
import os
import numpy as np
import h5py
import bokehplot as bkp
from data_processing import preprocess

def plotter(algo, **kw):
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')

    assert len(store_in.keys()) == 1
    data, _, _ = preprocess(
        data=store_in['data'],
        nbins_phi=kw['NbinsPhi'],
        nbins_rz=kw['NbinsRz'],
        window_size=kw['WindowSize']
    )

    plot_name = os.path.join('plots', 'ntriggercells.html')
    b = bkp.BokehPlot(plot_name, nfigs=len(data))

    for i,rzslice in enumerate(data):
        bins = rzslice[:,1].astype('int64')
        bincounts = np.bincount(bins)

        # b.graph(idx=0, data=[np.arange(len(bincounts)),bincounts],
        #         style='circle', color='orange', line=False)
        b.histogram( idx=i, data=np.histogram(bincounts, bins=20),
                     color='orange' )
    b.save_frame(show=True)


if __name__ == "__main__":
    from airflow.airflow_dag import optimization_kwargs
    for falgo in optimization_kwargs['FesAlgos']:
        plotter( falgo, **optimization_kwargs )
