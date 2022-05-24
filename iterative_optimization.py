import os
import csv
import argparse
from tqdm import tqdm
import random; random.seed(10)
from copy import copy
import numpy as np
import pandas as pd
import uproot as up
import h5py
import sys
#np.set_printoptions(threshold=sys.maxsize, linewidth=170)

from random_utils import get_html_name
from data_processing import DataProcessing
from plotter import Plotter

from airflow.airflow_dag import (
    optimization_kwargs as opt_kw,
    filling_kwargs,
    smoothing_kwargs,
    seeding_kwargs,
    clustering_kwargs,
    validation_kwargs,
    fill_path,
    )
from filling import filling
from smoothing import smoothing
from seeding import seeding
from clustering import clustering
from validation import validation, stats_collector
from plotting.trigger_cells_occupancy import (
    plot_trigger_cells_occupancy as plot_tc_occ
    )

def is_sorted(arr, nbinsphi):
    diff = arr[:-1] - arr[1:]
    return np.all( (diff<=0) | (diff==nbinsphi-1))

def process_trigger_cell_geometry_data(region, selection,
                                       positive_endcap_only=True, debug=False, **kw):
    """Prepare trigger cell geometry data to be used as
    input to the iterative algorithm."""
    tcDataPath = os.path.join(os.environ['PWD'], 'data', 'test_triggergeom.root')
    tcFile = up.open(tcDataPath)

    tcFolder = 'hgcaltriggergeomtester'
    tcTreeName = 'TreeTriggerCells'
    tcTree = tcFile[ os.path.join(tcFolder, tcTreeName) ]
    if debug:
        print('Input Tree:')
        print(tcTree.show())
        quit()

    simDataPath = os.path.join(os.environ['PWD'], 'data', 'gen_cl3d_tc.hdf5')
    simAlgoDFs, simAlgoFiles, simAlgoPlots = ({} for _ in range(3))
    fes = ['ThresholdDummyHistomaxnoareath20']
    for fe in fes:
        simAlgoFiles[fe] = [ os.path.join(simDataPath) ]

    tcVariables = {'zside', 'subdet', 'layer', 'phi', 'eta', 'x', 'y', 'z', 'id'}
    assert(tcVariables.issubset(tcTree.keys()))
    tcVariables = list(tcVariables)

    tcData = tcTree.arrays(tcVariables, library='pd')
    if debug:
        print( tcData.describe() )
    
    if positive_endcap_only:
        tcData = tcData[ tcData.zside == 1 ] #only look at positive endcap
        tcData = tcData.drop(['zside'], axis=1)
        tcVariables.remove('zside')

    # ECAL (1), HCAL silicon (2) and HCAL scintillator (10, here ignored)
    if region == 'Si':
        subdetCond = (tcData.subdet == 1) | (tcData.subdet == 2)
    elif region == 'ECAL':
        subdetCond = (tcData.subdet == 1)
    elif region == 'MaxShower':
        subdetCond = (tcData.subdet == 1) & (tcData.layer >= 8) & (tcData.layer<=15)

    tcData = tcData[ subdetCond ]
    tcData = tcData.drop(['subdet'], axis=1)
    tcVariables.remove('subdet')

    tcData['R'] = np.sqrt(tcData.x*tcData.x + tcData.y*tcData.y)
    tcData['Rz'] = tcData.R / abs(tcData.z)
    #the following cut removes almost no event at all
    tcData = tcData[ (tcData['Rz'] < kw['MaxROverZ']) & (tcData['Rz'] >  kw['MinROverZ']) ]
    
    copt = dict(labels=False)
    tcData['Rz_bin'] = pd.cut( tcData['Rz'], bins=kw['RzBinEdges'], **copt )
    tcData['phi_bin'] = pd.cut( tcData['phi'], bins=kw['PhiBinEdges'], **copt )

    # save data for optimization task
    inoptfile = fill_path(kw['OptimizationIn'],
                          selection=selection, region=region)

    with h5py.File(inoptfile, mode='w') as store:
        save_cols = ['R', 'Rz', 'phi', 'Rz_bin', 'phi_bin', 'id']
        saveData = ( tcData[save_cols]
                    .sort_values(by=['Rz_bin', 'phi'])
                    .to_numpy() )

        store['data'] = saveData
        store['data'].attrs['columns'] = save_cols
        doc = 'Trigger cell phi vs. R/z positions for optimization.'
        store['data'].attrs['doc'] = doc

def optimization(pars, **kw):
    outresen = fill_path(kw['OptimizationIn'],
                         selection=pars['selection'],
                         region=pars['region'])

    store_in  = h5py.File(outresen,  mode='r')
    plotter = Plotter(**opt_kw)
    mode = 'variance'
    window_size = 3

    assert len(store_in.keys()) == 1
    dp = DataProcessing( phi_bounds=(kw['MinPhi'],kw['MaxPhi']),
                         bin_bounds=(0,50) )

    # check detids make sense
    assert (np.sort(np.unique(store_in['data'][:,-1])) == np.sort(store_in['data'][:,-1])).all()

    data, bins, _, _, idx_d = dp.preprocess( data=store_in['data'],
                                             nbins_phi=kw['NbinsPhi'],
                                             nbins_rz=kw['NbinsRz'],
                                             window_size=window_size,
                                             normalize=False )

    store_in.close()

    def get_edge(idx, misalignment, ncellstot):
        """
        Returns the index corresponding to the first element in bin with id `id`
        """
        edge = sum(lb[:idx]) + misalignment
        if edge > ncellstot:
            edge -= ncellstot
        elif edge < 0:
            edge += ncellstot
        return edge
    
    for rzslice,(ldata,lbins) in enumerate(zip(data, bins)):
        ld = np.array(ldata)
        lb = np.array(lbins)
        radiae = ld[:,idx_d.r]
        phi_old = ld[:,idx_d.phi]

        run_algorithm = True
        if rzslice not in kw['LayersToOptimize']:
            run_algorithm = False
   
        if run_algorithm:
            plotter.reset()
             
            boundshift = window_size - 1
            ncellstot = sum(lb)
            lastidx = kw['NbinsPhi']-1
             
            plotter.save_orig_data( data=copy(lb), data_type='bins',
                                    boundary_sizes=0 )
             
            # initial differences for stopping criterion
            lb_orig2 = lb[:]
            lb_orig1 = np.roll(lb_orig2, +1)
            lb_orig3 = np.roll(lb_orig2, -1)
             
            gl_orig = lb_orig2 - lb_orig1
            gr_orig = lb_orig3 - lb_orig2
            stop = pars['iter_par'] * (abs(gl_orig) + abs(gr_orig))
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
                        ld[edge,idx_d.phibin] = id2
                        if id2==0:
                            misalign -= 1
             
                    elif side == 'right' and region in ('valley', 'ascent'):
                        edge = get_edge(id3, misalign, ncellstot)
                        lb[id3] -= 1
                        lb[id2] += 1
                        ld[edge,idx_d.phibin] = id2
                        if id2==lastidx:
                            misalign += 1
                
                    elif side == 'left' and region in ('mountain', 'ascent'):
                        edge = get_edge(id2, misalign, ncellstot)
                        lb[id1] += 1
                        lb[id2] -= 1
             
                        #SO DIRTY!!!!!!!!! Probably some very rare boundary condition issue.
                        try:
                            ld[edge,idx_d.phibin] = id1
                        except IndexError:
                            ld[edge-1,idx_d.phibin] = id1
                            
                        if id2==0:
                            misalign += 1
             
                    elif side == 'right' and region in ('mountain', 'descent'):
                        edge = get_edge(id3, misalign, ncellstot) - 1
                        lb[id3] += 1
                        lb[id2] -= 1
                        ld[edge,idx_d.phibin] = id3
                        if id2==lastidx:
                            misalign -= 1
                    else:
                        raise RuntimeError('Impossible 2!')                    

                    if not is_sorted(ld[:,idx_d.phibin], kw['NbinsPhi']):
                        print('Not Sorted!!!!!')
                        quit()
             
            phi_new_low_edges = kw['PhiBinEdges'][:-1][ld[:,idx_d.phibin].astype(int)]
            phi_new_high_edges = kw['PhiBinEdges'][1:][ld[:,idx_d.phibin].astype(int)]
             
            df = pd.DataFrame(dict(phi_old=phi_old,
                                   bin_old=np.array(data[rzslice])[:,idx_d.phibin],
                                   bin_new=ld[:,idx_d.phibin],
                                   # fix upstream inconsistency when producing ROOT files
                                   # needed for TC id comparison later on
                                   radius=radiae,
                                   id=np.uint32(ld[:,idx_d.tc_id])))
    
            # the bin edge to use to calculate the phi distance to the nearest edge depends on whether the trigger cell is moving
            # to the left or to the right bin. The following introduces a mask to perform the conditional decision.
            df['move_to_the_left'] = np.sign(df.bin_old - df.bin_new).astype(int)
            df['move_to_the_right'] = 0
            df.loc[ df.move_to_the_left == -1, 'move_to_the_right' ] = 1
            df.loc[ df.move_to_the_left == -1, 'move_to_the_left' ] = 0
            # each row must have either left or right equal to zero
            assert not np.count_nonzero(df.move_to_the_left * df.move_to_the_right != 0.) 
             
            # fix boundary conditions
            df.loc[ df.bin_old-df.bin_new == kw['NbinsPhi']-1,    'move_to_the_left']   = 0
            df.loc[ df.bin_old-df.bin_new == kw['NbinsPhi']-1,    'move_to_the_right']  = 1
            df.loc[ df.bin_old-df.bin_new == -(kw['NbinsPhi']-1), 'move_to_the_left']   = 1
            df.loc[ df.bin_old-df.bin_new == -(kw['NbinsPhi']-1), 'move_to_the_right']  = 0
             
            half_bin_width = 0.0001#(kw['PhiBinEdges'][1]-kw['PhiBinEdges'][0])/2
            #assert round(half_bin_width,5) == round((kw['PhiBinEdges'][-1]-kw['PhiBinEdges'][-2])/2,5)
            df['d_left']  = df.move_to_the_left  * abs(phi_old - phi_new_high_edges + half_bin_width)
            df['d_right'] = df.move_to_the_right * abs(phi_new_low_edges  - phi_old + half_bin_width)
            assert not np.count_nonzero(df.d_left * df.d_right != 0.) 
             
            df['distance'] = -1*df.d_left + df.d_right
             
            # the distance is zero when the bin does not change
            df.loc[ df.bin_old==df.bin_new, 'distance' ] = 0.
             
            nonzero_ratio = 1. - float(len(df[df.distance == 0])) / float(len(df.distance))
             
            # remove migrations in boundary conditions to avoid visualization issues
            df.loc[ df.distance >= np.pi, 'distance' ] = abs(df.loc[ df.distance >= np.pi, 'distance' ] - 2*np.pi)
            df.loc[ df.distance < -np.pi, 'distance' ] = -abs(df.loc[ df.distance < -np.pi, 'distance' ] + 2*np.pi)
             
            df['phi_new'] = df.phi_old + df.distance
             
            cond1 = (df.bin_new-df.bin_old < 0) & (df.distance>0)
            df.loc[cond1, 'phi_new'] = df.loc[cond1, 'phi_new'] - 2*np.pi
             
            cond2 = (df.bin_new-df.bin_old > 0) & (df.distance<0)
            df.loc[cond2, 'phi_new'] = df.loc[cond2, 'phi_new'] + 2*np.pi

            xdist_f = lambda i,f: df.radius*(np.cos(i)-np.cos(f))
            ydist_f = lambda i,f: df.radius*(np.sin(i)-np.sin(f))
            df['xdist'] = xdist_f(df.phi_old, df.phi_new)
            df['ydist'] = ydist_f(df.phi_old, df.phi_new)
            
            # boundary condition: from bin 0 to 215
            cond3 = ( (df.phi_new < 0.) & (df.phi_old > 0.)
                     & (df.phi_old-df.phi_new>kw['MaxPhi']) )
            df.loc[cond3, 'xdist'] = xdist_f(df.phi_old, df.phi_new+2*np.pi)
            df.loc[cond3, 'ydist'] = ydist_f(df.phi_old, df.phi_new+2*np.pi)

            # boundary condition: from bin 215 to 0
            cond4 = ( (df.phi_new > 0) & (df.phi_old < 0)
                     & (df.phi_old-df.phi_new<kw['MaxPhi']) )
            df.loc[cond4, 'xdist'] = xdist_f(df.phi_old, df.phi_new-2*np.pi)
            df.loc[cond4, 'ydist'] = ydist_f(df.phi_old, df.phi_new-2*np.pi)

            eucl_dist = np.sqrt(df.xdist**2+df.ydist**2)

            arcdist_f = lambda i,f: df.radius*(np.abs(i-f))
            df['arc'] = arcdist_f(df.phi_new, df.phi_old)
            df.loc[cond3, 'arc'] = arcdist_f(df.phi_old, df.phi_new+2*np.pi)
            df.loc[cond3, 'arc'] = arcdist_f(df.phi_old, df.phi_new+2*np.pi)
            df.loc[cond4, 'arc'] = arcdist_f(df.phi_old, df.phi_new-2*np.pi)
            df.loc[cond4, 'arc'] = arcdist_f(df.phi_old, df.phi_new-2*np.pi)

            plotter.save_gen_data(lb, boundary_sizes=0, data_type='bins')
            plotter.save_phi_distances(phi_dist=df.distance,
                                       eucl_dist=eucl_dist,
                                       arc_dist=df.arc)
            plotter.save_iterative_phi_tab(nonzero_ratio=nonzero_ratio,
                                           ncellstot=ncellstot )
            plotter.save_iterative_bin_tab()

        else: # if run_algorithm:
            df = pd.DataFrame(dict(phi_old=phi_old,
                                   phi_new=phi_old,
                                   radius=radiae,
                                   id=np.uint32(ld[:,idx_d.tc_id])))

        df = df[['phi_old', 'phi_new', 'id']]
        df_total = df if rzslice==0 else pd.concat((df_total,df), axis=0)

        # end loop over the layers
    plot_name = os.path.join( 'out',
                              get_html_name(__file__, extra='_'+str(pars['iter_par']).replace('.','p')) )
    plotter.plot_iterative( plot_name=plot_name,
                            tab_names = [''+str(x) for x in range(len(ldata))],
                            show_html=False )

    return df_total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--process',
                        help='reprocess trigger cell geometry data',
                        action='store_true')
    parser.add_argument('-p', '--plot',
                        help='plot shifted trigger cells instead of originals',
                        action='store_true')
    fill_help = ( 'Whether to filling the datasets or reuse the existing ones.' +
                  ' This step is by far the sowest in the chain.' )
    parser.add_argument('-f', '--do_filling',
                        help=fill_help,
                        action='store_false')
    parser.add_argument('-m', '--iter_par',
                        help='iterative algorithm tunable parameter',
                        default=0.5, type=float)
    parser.add_argument('-s', '--selection',
                        help='selection used to select cluster under study',
                        default='splits_only', type=str)
    parser.add_argument('-n', '--nevents',
                        help='selection used to select cluster under study',
                        default=-1, type=int)
    region_help = 'Z region in the detector considered for the trigger cell geometry.'
    parser.add_argument('--region',
                        help=region_help,
                        choices=('Si', 'ECAL', 'MaxShower'),
                        default='Si', type=str)
    FLAGS = parser.parse_args()

    if FLAGS.process:
        process_trigger_cell_geometry_data(region=FLAGS.region,
                                           selection=FLAGS.selection, **opt_kw )

    pars_d = {'iter_par': FLAGS.iter_par,
              'selection': FLAGS.selection,
              'region': FLAGS.region }
    outresen  = fill_path(opt_kw['OptimizationEnResOut'],  **pars_d)
    outrespos = fill_path(opt_kw['OptimizationPosResOut'], **pars_d)
    outcsv    = fill_path(opt_kw['OptimizationCSVOut'],    extension='csv', **pars_d)

    print('Starting iterative parameter {}.'.format(FLAGS.iter_par),
          flush=True)
    
    with open(outcsv, 'w', newline='') as csvfile, pd.HDFStore(outresen, mode='w') as storeEnRes, pd.HDFStore(outrespos, mode='w') as storePosRes:

        fieldnames = ['iter_par', 'c_loc1', 'c_loc2', 'c_rem1', 'c_rem2',
                      'locrat1', 'locrat2', 'remrat1', 'remrat2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        sys.stderr.flush()

        tc_map = optimization(pars_d, **opt_kw )

        if FLAGS.do_filling:
            filling(pars_d, FLAGS.nevents, tc_map, **filling_kwargs)

        smoothing(pars_d, **smoothing_kwargs)

        seeding(pars_d, **seeding_kwargs)

        clustering(pars_d, **clustering_kwargs)

        res = stats_collector(pars_d, **validation_kwargs)

        writer.writerow({fieldnames[0] : FLAGS.iter_par,
                         fieldnames[1] : res[0],
                         fieldnames[2] : res[1],
                         fieldnames[3] : res[2],
                         fieldnames[4] : res[3],
                         fieldnames[5] : res[4],
                         fieldnames[6] : res[5],
                         fieldnames[7] : res[6],
                         fieldnames[8] : res[7]})
            
        assert len(opt_kw['FesAlgos']) == 1
            
        df_enres = pd.DataFrame({'enres_old': res[8],
                                 'enres_new': res[9]})
        df_posres = pd.DataFrame({'etares_old': res[10],
                                  'etares_new': res[11],
                                  'phires_old': res[12],  
                                  'phires_new': res[13]})
        key = opt_kw['FesAlgos'][0] + '_data'

        storeEnRes [key] = df_enres
        storePosRes[key] = df_posres
        
        if FLAGS.plot:
            this_file = os.path.basename(__file__).split('.')[0]
            plot_name = fill_path(this_file,
                                  extension='html',
                                  **pars_d)
                
            plot_tc_occ(pars_d,
                        trigger_cell_map=tc_map,
                        plot_name=plot_name,
                        pos_endcap=True,
                        nevents=200,
                        min_rz=opt_kw['MinROverZ'],
                        max_rz=opt_kw['MaxROverZ'],
                        layer_edges=[0,28],
                        **opt_kw)

    print('Finished for iterative parameter {}.'.format(FLAGS.iter_par), flush=True)
