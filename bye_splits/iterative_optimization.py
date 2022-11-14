# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import tasks
import utils
from utils import params, common, parsing

from plot.trigger_cells_occupancy import plot_trigger_cells_occupancy

import csv
import argparse
import random; random.seed(10)
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import uproot as up
import h5py
import sys
import re
import itertools
#np.set_printoptions(threshold=sys.maxsize, linewidth=170)

def is_sorted(arr, nbinsphi):
    diff = arr[:-1] - arr[1:]
    return np.all( (diff<=0) | (diff==nbinsphi-1))

def process_trigger_cell_geometry_data(region, selection,
                                       positive_endcap_only=True, debug=False, **kw):
    """Prepare trigger cell geometry data to be used as
    input to the iterative algorithm."""

    tcDataPath = Path(__file__).parent.absolute().parent / params.DataFolder / 'test_triggergeom.root'
    tcFile = up.open(tcDataPath)

    tcFolder = 'hgcaltriggergeomtester'
    tcTreeName = 'TreeTriggerCells'
    tcTree = tcFile[ os.path.join(tcFolder, tcTreeName) ]
    if debug:
        print('Input Tree:')
        print(tcTree.show())
        quit()

    simDataPath = kw['InFile']
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

    tcData, subdetCond = common.tc_base_selection(tcData, region=region, pos_endcap=True,
                                                  range_rz=(kw['MinROverZ'], kw['MaxROverZ']))

    copt = dict(labels=False)
    tcData['Rz_bin'] = pd.cut( tcData['Rz'], bins=kw['RzBinEdges'], **copt )
    tcData['phi_bin'] = pd.cut( tcData['phi'], bins=kw['PhiBinEdges'], **copt )

    tcData_main = tcData[ subdetCond ]
    tcData_inv  = tcData[ ~subdetCond ]

    # save data for optimization task
    which = re.split('gen_cl3d_tc_|_ThresholdDummy',kw['InFile'])[1]
    inoptfile = common.fill_path('{}_{}'.format(kw['OptIn'],which), sel=selection, reg=region)

    with h5py.File(inoptfile, mode='w') as store:
        save_cols = ['R', 'Rz', 'phi', 'Rz_bin', 'phi_bin', 'id']
        saveData_inv = ( tcData_inv[save_cols]
                         .sort_values(by=['Rz_bin', 'phi'])
                         .to_numpy() )
        saveData_main = ( tcData_main[save_cols]
                         .sort_values(by=['Rz_bin', 'phi'])
                         .to_numpy() )

        store['data_inv']  = saveData_inv
        store['data_main'] = saveData_main
        store['data_inv'].attrs['columns'] = save_cols
        store['data_main'].attrs['columns'] = save_cols
        doc_inv = 'Trigger cell phi vs. R/z positions for optimization: inverted phase space (relative to active algo phase space).'
        doc_main = 'Trigger cell phi vs. R/z positions for optimization: active algo phase space.'
        store['data_inv'].attrs['doc'] = doc_inv
        store['data_main'].attrs['doc'] = doc_main

def optimization(pars, **kw):
    outresen = common.fill_path(kw['OptIn'], sel=pars['sel'], reg=pars['reg'])
    store_in  = h5py.File(outresen, mode='r')
    plot_obj = utils.plotter.Plotter(**params.opt_kw)
    mode = 'variance'
    window_size = 3

    assert list(store_in.keys()) == ['data_inv', 'data_main']
    dp = utils.data_processing.DataProcessing(phi_bounds=(kw['MinPhi'],kw['MaxPhi']),
                                              bin_bounds=(0,50))

    # check detids make sense
    assert (np.sort(np.unique(store_in['data_inv'][:,-1])) == np.sort(store_in['data_inv'][:,-1])).all()
    assert (np.sort(np.unique(store_in['data_main'][:,-1])) == np.sort(store_in['data_main'][:,-1])).all()

    data_opt = dict(nbins_phi=kw['NbinsPhi'], nbins_rz=kw['NbinsRz'],
                    window_size=window_size, normalize=False)
    data_main, bins_main, _, _, idx_d_main = dp.preprocess( data=store_in['data_main'], **data_opt)
    if pars['reg'] != 'All':
        data_inv, bins_inv, _, _, idx_d_inv = dp.preprocess( data=store_in['data_inv'], **data_opt)

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

    count = 0
    for rzslice, (ldata_main, lbins_main) in enumerate(zip(data_main, bins_main)):
        ld = np.array(ldata_main)
        count += len(ld)
        lb = np.array(lbins_main)
        radiae = ld[:,idx_d_main.r]
        phi_old = ld[:,idx_d_main.phi]

        run_algorithm = True
        if rzslice not in kw['LayersToOptimize']:
            run_algorithm = False

        if run_algorithm:
            plot_obj.reset()

            boundshift = window_size - 1
            ncellstot = sum(lb)
            lastidx = kw['NbinsPhi']-1

            plot_obj.save_orig_data( data=copy(lb), data_type='bins',
                                    boundary_sizes=0 )

            # initial differences for stopping criterion
            lb_orig2 = lb[:]
            lb_orig1 = np.roll(lb_orig2, +1)
            lb_orig3 = np.roll(lb_orig2, -1)

            gl_orig = lb_orig2 - lb_orig1
            gr_orig = lb_orig3 - lb_orig2
            stop = pars['ipar'] * (abs(gl_orig) + abs(gr_orig))
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
                        ld[edge,idx_d_main.phibin] = id2
                        if id2==0:
                            misalign -= 1

                    elif side == 'right' and region in ('valley', 'ascent'):
                        edge = get_edge(id3, misalign, ncellstot)
                        lb[id3] -= 1
                        lb[id2] += 1
                        ld[edge,idx_d_main.phibin] = id2
                        if id2==lastidx:
                            misalign += 1

                    elif side == 'left' and region in ('mountain', 'ascent'):
                        edge = get_edge(id2, misalign, ncellstot)
                        lb[id1] += 1
                        lb[id2] -= 1

                        #SO DIRTY!!!!!!!!! Probably some very rare boundary condition issue.
                        try:
                            ld[edge,idx_d_main.phibin] = id1
                        except IndexError:
                            ld[edge-1,idx_d_main.phibin] = id1

                        if id2==0:
                            misalign += 1

                    elif side == 'right' and region in ('mountain', 'descent'):
                        edge = get_edge(id3, misalign, ncellstot) - 1
                        lb[id3] += 1
                        lb[id2] -= 1
                        ld[edge,idx_d_main.phibin] = id3
                        if id2==lastidx:
                            misalign -= 1
                    else:
                        raise RuntimeError('Impossible 2!')

                    if not is_sorted(ld[:,idx_d_main.phibin], kw['NbinsPhi']):
                        print('Not Sorted!!!!!')
                        quit()

            phi_new_low_edges = kw['PhiBinEdges'][:-1][ld[:,idx_d_main.phibin].astype(int)]
            phi_new_high_edges = kw['PhiBinEdges'][1:][ld[:,idx_d_main.phibin].astype(int)]

            df = pd.DataFrame(dict(phi_old=phi_old,
                                   bin_old=np.array(data_main[rzslice])[:,idx_d_main.phibin],
                                   bin_new=ld[:,idx_d_main.phibin],
                                   # fix upstream inconsistency when producing ROOT files
                                   # needed for TC id comparison later on
                                   radius=radiae,
                                   id=np.uint32(ld[:,idx_d_main.tc_id])))

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

            plot_obj.save_gen_data(lb, boundary_sizes=0, data_type='bins')
            plot_obj.save_phi_distances(phi_dist=df.distance,
                                       eucl_dist=eucl_dist,
                                       arc_dist=df.arc)
            plot_obj.save_iterative_phi_tab(nonzero_ratio=nonzero_ratio,
                                           ncellstot=ncellstot )
            plot_obj.save_iterative_bin_tab()

        else: # if run_algorithm:
            df = pd.DataFrame(dict(phi_old=phi_old,
                                   phi_new=phi_old,
                                   radius=radiae,
                                   id=np.uint32(ld[:,idx_d_main.tc_id])))

        df = df[['phi_old', 'phi_new', 'id']]

        df_total = df if rzslice==0 else pd.concat((df_total,df), axis=0)

    if pars['reg'] != 'All':
        for rzslice, (ldata_inv, lbins_inv) in enumerate(zip(data_inv, bins_inv)):
            ld_inv = np.array(ldata_inv)
            lb_inv = np.array(ldata_inv)
            phi_inv = ld_inv[:,idx_d_inv.phi]

            df_inv = pd.DataFrame(dict(phi=phi_inv,
                                        id=np.uint32(ld_inv[:,idx_d_inv.tc_id])))
            df_inv_total = df_inv if rzslice==0 else pd.concat((df_inv_total,df_inv), axis=0)

        # end loop over the layers

    plot_name = common.get_html_name(__file__, name='plot_'+str(pars['ipar']).replace('.','p'))
    plot_obj.plot_iterative(plot_name=plot_name,
                            tab_names = [''+str(x) for x in range(len(ldata_main))],
                            show_html=False)

    if pars['reg'] != 'All':
        assert not set(df_total.id) & set(df_inv_total.id)
        df_merge = df_inv_total.merge(df_total, how='outer', on='id')
        not_null = df_merge.phi.notnull()
        not_null_phis = df_merge.loc[df_merge.phi.notnull() == True, 'phi']
        df_merge.loc[not_null, 'phi_new'] = not_null_phis
        df_merge.loc[not_null, 'phi_old'] = not_null_phis

        df_merge = df_merge.drop(['phi'], axis=1)
    else:
        df_merge = df_total

    return df_merge

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-r', '--process',
                        help='reprocess trigger cell geometry data',
                        action='store_true')
    parser.add_argument('-p', '--plot',
                        help='plot shifted trigger cells instead of originals',
                        action='store_true')
    parser.add_argument('--no_fill',    action='store_true')
    parser.add_argument('--no_smooth',  action='store_true')
    parser.add_argument('--no_seed',    action='store_true')
    parser.add_argument('--no_cluster', action='store_true')
    nevents_help = "Number of events for processing. Pass '-1' for all events."
    parser.add_argument('-n', '--nevents', help=nevents_help,
                        default=-1, type=int)
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()
    assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

    #input_files = [re.split('pion_',file)[1] for file in params.fill_kw['FillInFiles']['pion']]
    input_files = params.fill_kw['FillInFiles']
    #input_files = [['{}_{}{}'.format(re.split(r'(gen_cl3d_tc)',file)[1],params.base_kw['FesAlgos'][0],re.split(r'(gen_cl3d_tc)',file)[2]) for file in input_files[key]] for key in input_files.keys()]
    #input_files = ['{}_{}{}'.format(re.split(r'(gen_cl3d_tc)',f)[1],params.base_kw['FesAlgos'][0],re.split(r'(gen_cl3d_tc)',f)[2]) for f in input_files]
    simDataPaths = [[os.path.join(params.base_kw['BasePath'], infile) for infile in input_files[key]] for key in input_files.keys()]
    simDataPaths = list(itertools.chain(*simDataPaths))

    if FLAGS.process:
        for path in simDataPaths:
            params.opt_kw['InFile'] = path
            process_trigger_cell_geometry_data(region=FLAGS.reg,
                                           selection=FLAGS.sel, **params.opt_kw)

    pars_d = {'sel'           : FLAGS.sel,
              'reg'           : FLAGS.reg,
              'seed_window'   : FLAGS.seed_window,
              'smooth_kernel' : FLAGS.smooth_kernel,
              'cluster_algo'  : FLAGS.cluster_algo }
    pars_d.update({'ipar': FLAGS.ipar})

    print('Starting iterative parameter {}.'.format(FLAGS.ipar),
          flush=True)

    for path in simDataPaths:
        # Get file addition
        file = re.split('gen_cl3d_tc_|_ThresholdDummy',path)[1]
        outcsv = common.fill_path('{}_{}'.format(params.opt_kw['OptCSVOut'],file), ext='csv', **pars_d)
        outresen  = common.fill_path('{}_{}'.format(params.opt_kw['OptEnResOut'],file),  **pars_d)
        outrespos = common.fill_path('{}_{}'.format(params.opt_kw['OptPosResOut'],file), **pars_d)

        with open(outcsv, 'w', newline='') as csvfile, pd.HDFStore(outresen, mode='w') as storeEnRes, pd.HDFStore(outrespos, mode='w') as storePosRes:
            fieldnames = ['ipar', 'c_loc1', 'c_loc2', 'c_rem1', 'c_rem2',
                          'locrat1', 'locrat2', 'remrat1', 'remrat2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            sys.stderr.flush()

            # Set file specific parameters
            file_pars = common.dict_per_file(params,path)

            tc_map = optimization(pars_d, **file_pars['opt'])

            if not FLAGS.no_fill:
                tasks.fill.fill(pars_d, FLAGS.nevents, tc_map, **file_pars['fill'])

            if not FLAGS.no_smooth:
                tasks.smooth.smooth(pars_d, **file_pars['smooth'])

            if not FLAGS.no_seed:
                tasks.seed.seed(pars_d, **file_pars['seed'])

            if not FLAGS.no_cluster:
                tasks.cluster.cluster(pars_d, **file_pars['cluster'])

            # Validation currently failing (specifically line 202 of tasks/validation.py)
            '''res = tasks.validation.stats_collector(pars_d, **file_pars['validation'])

            writer.writerow({fieldnames[0] : FLAGS.ipar,
                             fieldnames[1] : res[0],
                             fieldnames[2] : res[1],
                             fieldnames[3] : res[2],
                             fieldnames[4] : res[3],
                             fieldnames[5] : res[4],
                             fieldnames[6] : res[5],
                             fieldnames[7] : res[6],
                             fieldnames[8] : res[7]})

            assert len(params.opt_kw['FesAlgos']) == 1 # Be wary

            df_enres = pd.DataFrame({'enres_old': res[8],
                                     'enres_new': res[9]})
            df_posres = pd.DataFrame({'etares_old': res[10],
                                      'etares_new': res[11],
                                      'phires_old': res[12],
                                      'phires_new': res[13]})
            key = params.opt_kw['FesAlgos'][0] + '_data'

            storeEnRes [key] = df_enres
            storePosRes[key] = df_posres'''

            if FLAGS.plot:
                this_file = os.path.basename(__file__).split('.')[0]
                plot_name = common.fill_path(this_file, ext='html', **pars_d)
                print(plot_name)
                plot_trigger_cells_occupancy(pars_d,
                                             plot_name=plot_name,
                                             pos_endcap=True,
                                             layer_edges=[0,42],
                                             nevents=1,
                                             min_rz=params.opt_kw['MinROverZ'],
                                             max_rz=params.opt_kw['MaxROverZ'],
                                             **params.opt_kw)

    print('Finished for iterative parameter {}.'.format(FLAGS.ipar), flush=True)
