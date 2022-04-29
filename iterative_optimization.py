import os
import random
import numpy as np
import h5py

from data_processing import DataProcessing
from plotter import Plotter
                
def optimization(**kw):
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    plotter = Plotter(**optimization_kwargs)
    mode = 'variance'
    window_size = 3 # no boundaries added

    assert len(store_in.keys()) == 1
    dp = DataProcessing( phi_bounds=(kw['MinPhi'],kw['MaxPhi']),
                         bin_bounds=(0,50) )
    data, bins, _, _ = dp.preprocess( data=store_in['data'],
                                      nbins_phi=kw['NbinsPhi'],
                                      nbins_rz=kw['NbinsRz'],
                                      window_size=window_size,
                                      normalize=False )
    chosen_layer = 0
  
    plotter.save_orig_data( data=np.array(data[chosen_layer][:,0]),
                            data_type='data',
                            boundary_sizes=0 )
    plotter.save_orig_data( data=np.array(bins),
                            data_type='bins',
                            boundary_sizes=0 )

    def get_edge(idx, misalignment):
        """returns the index corresponding to the first element in bin with id `id`"""
        edge = sum(lb[:idx]) + misalignment
        if edge > kw['NbinsPhi']:
            edge -= kw['NbinsPhi']
        elif edge < 0:
            edge += kw['NbinsPhi']
        return edge
    
    for i,(ldata,lbins) in enumerate(zip(data, bins)):
        if i!=chosen_layer:  #look at the first R/z slice only
            continue

        boundshift = window_size - 1
        ld = np.array(ldata)
        lb = np.array(lbins)
        ncellstot = sum(lb)
        lastidx = kw['NbinsPhi']-1

        # initial differences for stopping criterion
        lb_orig2 = lb[:]
        lb_orig1 = np.roll(lb_orig2, +1)
        lb_orig3 = np.roll(lb_orig2, -1)
        gl_orig = lb_orig2 - lb_orig1
        gr_orig = lb_orig3 - lb_orig2
        stop = 0.7 * (abs(gl_orig) + abs(gr_orig))

        idxs = [ np.arange(kw['NbinsPhi']) ]
        for _ in range(boundshift):
            idxs.append( np.roll(idxs[-1], -1) )

        # "excess" (positive or negative): how much the first bin 0 cell is misaligned
        # with respect to its starting position
        # required due to the cyclic boundary conditions
        misalign = 0

        print(lb_orig2, sum(lb_orig2))
        print()
        at_least_one = True
        while at_least_one:
            at_least_one = False
            for id_tuple in zip(*idxs):
                # triplet bin indices
                id1, id2, id3 = id_tuple

                # bin counts
                c1, c2, c3 = lb[id1], lb[id2], lb[id3]

                # left and right gradients
                gl = c2 - c1
                gr = c3 - c2
                gsum = abs(gl) + abs(gr)

                # stopping criterion
                # must be satisfied for all triplets
                if gsum < stop[id2]:
                    continue
                at_least_one = True
                
                # weights for random draw
                wl = abs(gl) / gsum
                assert( 0. <= wl <= 1.)
                wr = abs(gr) / gsum
                assert( 0. <= wr <= 1.)

                # "region" based on left and right gradients
                if gl <= 0 and gr >= 0:
                    region = 'valley'
                elif gl >= 0 and gr <= 0:
                    region = 'mountain'
                elif gl >= 0 and gr >= 0:
                    region = 'ascent'
                elif gl <= 0 and gr <= 0:
                    region = 'descent'
                else:
                    raise RuntimeError('Impossible 1!')
                
                # random draw (pick a side)
                side = 'left' if random.random() < wl else 'right'

                if side == 'left' and region in ('valley', 'descent'):
                    edge = get_edge(id2, misalign) - 1
                    lb[id1] -= 1
                    lb[id2] += 1
                    ld[edge,1] = id2
                    if id2==lastidx:
                        misalign -= 1

                elif side == 'right' and region in ('valley', 'ascent'):
                    edge = get_edge(id3, misalign)
                    lb[id3] -= 1
                    lb[id2] += 1
                    ld[edge,1] = id2
                    if id2==lastidx:
                        misalign += 1

                elif side == 'left' and region in ('mountain', 'ascent'):
                    edge = get_edge(id2, misalign)
                    lb[id1] += 1
                    lb[id2] -= 1
                    ld[edge,1] = id1
                    if id2==0:
                        misalign += 1

                elif side == 'right' and region in ('mountain', 'descent'):
                    edge = get_edge(id3, misalign) - 1
                    lb[id3] += 1
                    lb[id2] -= 1
                    ld[edge,1] = id3
                    if id2==lastidx:
                        misalign -= 1
                    
                else:
                    raise RuntimeError('Impossible 2!')                    

        print(lb, sum(lb))
            
        # plotter.save_gen_data(outdata.numpy(), boundary_sizes=0, data_type='data')
        # plotter.save_gen_data(outbins.numpy(), boundary_sizes=0, data_type='bins')

        # plotter.plot(plot_name=plot_name,
        #              minval=-1, maxval=52,
        #              density=False, show_html=False)

if __name__ == "__main__":  
    Nevents = 16#{{ dag_run.conf.nevents }}
    NbinsRz = 42
    NbinsPhi = 216
    MinROverZ = 0.076
    MaxROverZ = 0.58
    MinPhi = -np.pi
    MaxPhi = +np.pi
    DataFolder = 'data'
    optimization_kwargs = { 'NbinsRz': NbinsRz,
                            'NbinsPhi': NbinsPhi,
                            'MinROverZ': MinROverZ,
                            'MaxROverZ': MaxROverZ,
                            'MinPhi': MinPhi,
                            'MaxPhi': MaxPhi,

                            'LayerEdges': [0,28],
                            'IsHCAL': False,

                            'Debug': True,
                            'DataFolder': DataFolder,
                            'FesAlgos': ['ThresholdDummyHistomaxnoareath20'],
                            'BasePath': os.path.join(os.environ['PWD'], DataFolder),

                            'RzBinEdges': np.linspace( MinROverZ, MaxROverZ, num=NbinsRz+1 ),
                            'PhiBinEdges': np.linspace( MinPhi, MaxPhi, num=NbinsPhi+1 ),
                            'OptimizationIn': os.path.join(os.environ['PWD'], DataFolder, 'triggergeom_condensed.hdf5'),
                            'OptimizationOut': 'None.hdf5',
                           }

    optimization( **optimization_kwargs )
