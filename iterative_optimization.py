import os
import random; random.seed(10)
import numpy as np
import pandas as pd
import h5py
import sys; np.set_printoptions(threshold=sys.maxsize, linewidth=170)

from random_utils import get_html_name
from data_processing import DataProcessing
from plotter import Plotter
from copy import copy

from airflow.airflow_dag import (
    optimization_kwargs,
    filling_kwargs,
    smoothing_kwargs,
    seeding_kwargs,
    clustering_kwargs,
    validation_kwargs
    )
from filling import filling
from smoothing import smoothing
from seeding import seeding
from clustering import clustering
from validation import validation

def is_sorted(arr, nbinsphi):
    diff = arr[:-1] - arr[1:]
    return np.all( (diff==0) | (diff==-1) | (diff==-2) | (diff==nbinsphi-1))

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

    def get_edge(idx, misalignment, ncellstot):
        """returns the index corresponding to the first element in bin with id `id`"""
        edge = sum(lb[:idx]) + misalignment
        if edge > ncellstot:
            edge -= ncellstot
        elif edge < 0:
            edge += ncellstot
        return edge
    
    for ilayer,(ldata,lbins) in enumerate(zip(data, bins)):

        plotter.reset()
        
        boundshift = window_size - 1
        ld = np.array(ldata)
        lb = np.array(lbins)
        ncellstot = sum(lb)
        lastidx = kw['NbinsPhi']-1

        def get_edge(idx, misalignment):
            """returns the index corresponding to the first element in bin with id `id`"""
            edge = sum(lb[:idx]) + misalignment

            if edge >= ncellstot:
                edge -= ncellstot
            elif edge < 0:
                edge += ncellstot
            return edge

        # initial differences for stopping criterion
        lb_orig2 = lb[:]
        lb_orig1 = np.roll(lb_orig2, +1)
        lb_orig3 = np.roll(lb_orig2, -1)
        
        gl_orig = lb_orig2 - lb_orig1
        gr_orig = lb_orig3 - lb_orig2
        stop = 0.7 * (abs(gl_orig) + abs(gr_orig))
        stop[stop<1] = 1 # algorithm stabilisation

        idxs = [ np.arange(kw['NbinsPhi']) ]
        for _ in range(boundshift):
            idxs.append( np.roll(idxs[-1], -1) )

        # "excess" (positive or negative): how much the first bin 0 cell is misaligned
        # with respect to its starting position
        # required due to the cyclic boundary conditions
        misalign = 0

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
                if gsum <= stop[id2]:
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
                    edge = get_edge(id2, misalign, ncellstot) - 1
                    lb[id1] -= 1
                    lb[id2] += 1
                    ld[edge,1] = id2
                    if id2==0:
                        misalign -= 1

                elif side == 'right' and region in ('valley', 'ascent'):
                    edge = get_edge(id3, misalign, ncellstot)
                    lb[id3] -= 1
                    lb[id2] += 1
                    ld[edge,1] = id2
                    if id2==lastidx:
                        misalign += 1

                elif side == 'left' and region in ('mountain', 'ascent'):
                    edge = get_edge(id2, misalign, ncellstot)
                    lb[id1] += 1
                    lb[id2] -= 1
                    ld[edge,1] = id1
                    if id2==0:
                        misalign += 1

                elif side == 'right' and region in ('mountain', 'descent'):
                    edge = get_edge(id3, misalign, ncellstot) - 1
                    lb[id3] += 1
                    lb[id2] -= 1
                    ld[edge,1] = id3
                    if id2==lastidx:
                        misalign -= 1
                else:
                    raise RuntimeError('Impossible 2!')                    

                if not is_sorted(ld[:,1], kw['NbinsPhi']):
                    print('Not Sorted!!!!!')
                    quit()
                    # print('Edge: {}'.format(edge))
                    # print("Side: {}, Region: {}, Ids: {}, Misalign: {}".format(side, (id1,id2,id3), region, misalign))
                    # print(ld[:,1])
                    # breakpoint()

        phi_new_low_edges = kw['PhiBinEdges'][:-1][ld[:,1].astype(int)]
        phi_new_high_edges = kw['PhiBinEdges'][1:][ld[:,1].astype(int)]

        half_bin_width = 0.#(kw['PhiBinEdges'][1]-kw['PhiBinEdges'][0])/2
        df = pd.DataFrame(dict(phi_old=phi_old,
                               bin_old=np.array(data[ilayer])[:,1],
                               bin_new=ld[:,1]))

        # the bin edge to use to calculate the phi distance to the nearest edge depends on whether the trigger cell is moving
        # to the left or to the right bin. The following introduces a mask to perform the conditional decision.
        df['move_to_the_left'] = np.sign(df.bin_old - df.bin_new).astype(int)
        df['move_to_the_right'] = 0
        df.loc[ df.move_to_the_left == -1, 'move_to_the_right' ] = 1
        df.loc[ df.move_to_the_left == -1, 'move_to_the_left' ] = 0

        # fix boundary conditions
        df.loc[ df.bin_old-df.bin_new == kw['NbinsPhi']-1,    'move_to_the_left']   = 0
        df.loc[ df.bin_old-df.bin_new == kw['NbinsPhi']-1,    'move_to_the_right']  = 1
        df.loc[ df.bin_old-df.bin_new == -(kw['NbinsPhi']-1), 'move_to_the_left']   = 1
        df.loc[ df.bin_old-df.bin_new == -(kw['NbinsPhi']-1), 'move_to_the_right']  = 0

        print(df.bin_new)
        quit()

        df['distance'] = half_bin_width + df.move_to_the_left*abs(phi_new_high_edges-phi_old) + df.move_to_the_right*abs(phi_new_low_edges-phi_old)
        df['d_rigth'] = df.move_to_the_right*abs(phi_new_low_edges-phi_old)
        df['d_left'] = df.move_to_the_left*abs(phi_new_high_edges-phi_old)

        # the distance is zero when the bin did not change
        df.loc[ df.bin_old==df.bin_new, 'distance' ] = 0.

        nonzero_ratio = 1. - float(len(df[df.distance == 0])) / float(len(df.distance))
        df['phi_new'] = df.distance + df.phi_old

        # remove migrations in boundary conditions to avoid visualization issues
        df.loc[ df.distance > np.pi, 'distance' ] = abs( df.loc[ df.distance > np.pi, 'distance' ] - 2*np.pi )

        plotter.save_gen_data(lb, boundary_sizes=0, data_type='bins')
        plotter.save_gen_phi_data(df.distance)
        plotter.save_iterative_phi_tab(nonzero_ratio=nonzero_ratio,
                                       ncellstot=ncellstot )
        plotter.save_iterative_bin_tab()

        # for ilayer,(ldata,lbins) in enumerate(zip(data, bins)):
        # end loop over the layers
        
    plotter.plot_iterative( plot_name=get_html_name(__file__),
                           tab_names = [''+str(x) for x in range(len(ldata))],
                           show_html=False )

if __name__ == "__main__":  
    # Nevents = 16#{{ dag_run.conf.nevents }}
    # NbinsRz = 42
    # NbinsPhi = 216
    # MinROverZ = 0.076
    # MaxROverZ = 0.58
    # MinPhi = -np.pi
    # MaxPhi = +np.pi
    # DataFolder = 'data'
    # optimization_kwargs = { 'NbinsRz': NbinsRz,
    #                         'NbinsPhi': NbinsPhi,
    #                         'MinROverZ': MinROverZ,
    #                         'MaxROverZ': MaxROverZ,
    #                         'MinPhi': MinPhi,
    #                         'MaxPhi': MaxPhi,

    #                         'LayerEdges': [0,28],
    #                         'IsHCAL': False,

    #                         'Debug': True,
    #                         'DataFolder': DataFolder,
    #                         'FesAlgos': ['ThresholdDummyHistomaxnoareath20'],
    #                         'BasePath': os.path.join(os.environ['PWD'], DataFolder),

    #                         'RzBinEdges': np.linspace( MinROverZ, MaxROverZ, num=NbinsRz+1 ),
    #                         'PhiBinEdges': np.linspace( MinPhi, MaxPhi, num=NbinsPhi+1 ),
    #                         'OptimizationIn': os.path.join(os.environ['PWD'], DataFolder, 'triggergeom_condensed.hdf5'),
    #                         'OptimizationOut': 'None.hdf5',
    #                        }

    # tc_mapping = optimization( **optimization_kwargs )

    # filling    (**filling_kwargs, tc_mapping)
    filling    (**filling_kwargs)
    # smoothing  (**smoothing_kwargs)
    # seeding    (**seeding_kwargs)
    # clustering (**clustering_kwargs)
    # validating()
