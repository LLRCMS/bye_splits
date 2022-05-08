import os
import random; random.seed(18)
import argparse
import numpy as np
import pandas as pd
import uproot as up
import h5py
from bokeh.io import export_png

from bokeh.io import output_file, show
from bokeh.layouts import layout
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LogColorMapper, LogTicker,
                          LinearColorMapper, BasicTicker,
                          PrintfTickFormatter,
                          Range1d,
                          Panel, Tabs)
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.palettes import viridis as _palette

import sys
sys.path.append( os.environ['PWD'] )
from airflow.airflow_dag import filling_kwargs as kw

def calculateRoverZfromEta(eta):
    """R/z = arctan(theta) [theta is obtained from pseudo-rapidity, eta]"""
    _theta = 2*np.arctan( np.exp(-1 * eta) )
    return np.arctan( _theta )

def set_figure_props(p, xbincenters, ybincenters):
    """set figure properties"""
    p.axis.axis_line_color = 'black'
    p.axis.major_tick_line_color = 'black'
    p.axis.major_label_text_font_size = '10px'
    p.axis.major_label_standoff = 2
    p.xaxis.axis_label = r"$$\color{black} \phi$$"
    p.yaxis.axis_label = '$$R/z$$'
    
    p.hover.tooltips = [
        ("#hits", "@{'nhits'}"),
        ("min(eta)", "@{min_eta}"),
        ("max(eta)", "@{max_eta}"),
    ]

def plot_trigger_cells_occupancy(trigger_cell_map, pos_endcap,
                                 min_rz, max_rz,
                                 layer_edges,
                                 log_scale=False):
    rzBinCenters = ['{:.2f}'.format(x) for x in ( kw['RzBinEdges'][1:] + kw['RzBinEdges'][:-1] ) / 2 ]
    phiBinCenters = ['{:.2f}'.format(x) for x in ( kw['PhiBinEdges'][1:] + kw['PhiBinEdges'][:-1] ) / 2 ]
    binDistRz = kw['RzBinEdges'][1] - kw['RzBinEdges'][0] #assumes the binning is regular
    binDistPhi = kw['PhiBinEdges'][1] - kw['PhiBinEdges'][0] #assumes the binning is regular
    binConv = lambda vals,dist,amin : (vals*dist) + (dist/2) + amin

    SHIFTH, SHIFTV = binDistPhi, binDistRz

    tcDataPath = os.path.join(kw['BasePath'], 'test_triggergeom.root')
    tcFile = up.open(tcDataPath)

    tcFolder = 'hgcaltriggergeomtester'
    tcTreeName = 'TreeTriggerCells'
    tcTree = tcFile[ os.path.join(tcFolder, tcTreeName) ]

    simDataPath = os.path.join(os.environ['PWD'], 'data', 'gen_cl3d_tc.hdf5')
    simAlgoDFs, simAlgoFiles, simAlgoPlots = ({} for _ in range(3))
    fes = ['ThresholdDummyHistomaxnoareath20']
    for fe in fes:
        simAlgoFiles[fe] = [ os.path.join(simDataPath) ]

    title_common = r'{} vs {} bins'.format(kw['NbinsPhi'], kw['NbinsRz'])
    if pos_endcap:
        title_common += '; Positive end-cap only'
    title_common += '; Min(R/z)={} and Max(R/z)={}'.format(min_rz, max_rz)

    mypalette = _palette(50)
    #########################################################################
    ################### INPUTS: TRIGGER CELLS ###############################
    #########################################################################
    tcVariables = {'zside', 'subdet', 'layer', 'phi', 'eta', 'x', 'y', 'z', 'id'}
    assert(tcVariables.issubset(tcTree.keys()))
    tcVariables = list(tcVariables)

    tcData = tcTree.arrays(tcVariables, library='pd')
    # print( tcData.describe() )

    #########################################################################
    ################### INPUTS: SIMULATION 0 PU PHOTONS #####################
    #########################################################################
    for fe,files in simAlgoFiles.items():
        name = fe
        dfs = []
        for afile in files:
            with pd.HDFStore(afile, mode='r') as store:
                dfs.append(store[name])
        simAlgoDFs[fe] = pd.concat(dfs)

    simAlgoNames = sorted(simAlgoDFs.keys())

    #########################################################################
    ################### DATA ANALYSIS: TRIGGER CELLS ########################
    #########################################################################
    if pos_endcap:
        tcData = tcData[ tcData.zside == 1 ] #only look at positive endcap
        tcData = tcData.drop(['zside'], axis=1)
        tcVariables.remove('zside')

    # ignoring hgcal scintillator
    #subdetCond = tcData.subdet == 2 if flags.hcal else tcData.subdet == 1
    subdetCond = (tcData.subdet == 1) | (tcData.subdet == 2) #look at ECAL and HCAL
    tcData = tcData[ subdetCond ]
    tcData = tcData.drop(['subdet'], axis=1)
    tcVariables.remove('subdet')

    tcData['Rz'] = np.sqrt(tcData.x*tcData.x + tcData.y*tcData.y) / abs(tcData.z)
    #the following cut removes almost no event at all
    tcData = tcData[ (tcData['Rz'] < max_rz) & (tcData['Rz'] > min_rz) ]

    tcData.id = np.uint32(tcData.id)

    tcData = tcData.merge(trigger_cell_map, on='id', how='right').dropna()
    assert_diff = tcData.phi_old - tcData.phi
    assert not np.count_nonzero(assert_diff)

    copt = dict(labels=False)
    tcData['Rz_bin'] = pd.cut( tcData['Rz'], bins=kw['RzBinEdges'], **copt )

    # to check the effect of NOT applying the tc mapping
    # replace `phi_new` by `phi_old`
    tcData['phi_bin'] = pd.cut( tcData['phi_new'], bins=kw['PhiBinEdges'], **copt )
    
    # Convert bin ids back to values (central values in each bin)
    tcData['Rz_center'] = binConv(tcData['Rz_bin'], binDistRz, min_rz)
    tcData['phi_center'] = binConv(tcData['phi_bin'], binDistPhi, -np.pi)
    
    tcData.drop(['Rz_bin', 'phi_bin', 'Rz', 'phi'], axis=1, inplace=True)

    # if `-1` is included in layer_edges, the full selection is also drawn
    try:
        layer_edges.remove(-1)
        leftLayerEdges, rightLayerEdges = layer_edges[:-1], layer_edges[1:]
        leftLayerEdges.insert(0, 0)
        rightLayerEdges.insert(0, tcData.layer.max())
    except ValueError:
        leftLayerEdges, rightLayerEdges = layer_edges[:-1], layer_edges[1:]

    ledgeszip = tuple(zip(leftLayerEdges,rightLayerEdges))
    tcSelections = ['layer>{}, layer<={}'.format(x,y) for x,y in ledgeszip]
    groups = []
    for lmin,lmax in ledgeszip:
        groups.append( tcData[ (tcData.layer>lmin) & (tcData.layer<=lmax) ] )
        groupby = groups[-1].groupby(['Rz_center', 'phi_center'], as_index=False)
        groups[-1] = groupby.count()

        print(tcData)
        print(groups[-1])
        quit()

        eta_mins = groupby.min()['eta']
        eta_maxs = groupby.max()['eta']
        groups[-1].insert(0, 'min_eta', eta_mins)
        groups[-1].insert(0, 'max_eta', eta_maxs)
        groups[-1] = groups[-1].rename(columns={'z': 'nhits'})
        groups[-1] = groups[-1][['phi_center', 'nhits', 'Rz_center', 'min_eta', 'max_eta']]

    #########################################################################
    ################### DATA ANALYSIS: SIMULATION ###########################
    #########################################################################
    enresgrid = []
    enrescuts = [-0.35]
    assert(len(enrescuts)==len(fes))
    for i,(fe,cut) in enumerate(zip(fes,enrescuts)):
        df = simAlgoDFs[fe]
        #print(df.groupby("event").filter(lambda x: len(x) > 1))

        df = df[ (df['genpart_exeta']>1.7) & (df['genpart_exeta']<2.8) ]
        df = df[ df['cl3d_eta']>0 ]
        df['enres'] = ( df['cl3d_energy']-df['genpart_energy'] ) / df['genpart_energy']

        #### Energy resolution histogram ######################################################################
        hist, edges = np.histogram(df['enres'], density=True, bins=150)

        p = figure( width=500, height=300, title='Energy Resolution: ' + fe,
                    y_axis_type="log")
        virtualmin = 1e-4 #avoid log scale issues
        p.quad(top=hist, bottom=virtualmin, left=edges[:-1], right=edges[1:],
               fill_color="navy", line_color="white", alpha=0.7)
        p.line(x=[cut,cut], y=[virtualmin,max(hist)], line_color="#ff8888", line_width=4, alpha=0.9, legend_label="Cut")
        enresgrid.append(p)
        ######################################################################################################

        nansel = pd.isna(df['enres']) 
        nandf = df[nansel]
        nandf['enres'] = 1.1
        df = df[~nansel]
        df = pd.concat([df,nandf], sort=False)

        # select events with splitted clusters
        splittedClusters = df[ df['enres'] < cut ]

        # random pick some events (fixing the seed for reproducibility)
        _events_remaining = list(splittedClusters.index.unique())
        _events_sample = random.sample(_events_remaining, kw['Nevents'])
        splittedClusters = splittedClusters.loc[_events_sample]
        #splittedClusters.sample(n=NEVENTS, replace=False, random_state=8)

        #splitting remaining data into cluster and tc to avoid tc data duplication
        _cl3d_vars = [x for x in splittedClusters.columns.to_list() if 'tc_' not in x]
        splittedClusters_3d = splittedClusters[_cl3d_vars]
        splittedClusters_3d = splittedClusters_3d.reset_index()
        _tc_vars = [x for x in splittedClusters.columns.to_list() if 'cl3d' not in x]

        #trigger cells info is repeated across clusters in the same event
        splittedClusters_tc = splittedClusters.groupby("event").head(1)[_tc_vars] #first() instead of head(1) also works

        _tc_vars = [x for x in _tc_vars if 'tc_' in x]
        splittedClusters_tc = splittedClusters_tc.explode( _tc_vars )

        for v in _tc_vars:
            splittedClusters_tc[v] = splittedClusters_tc[v].astype(np.float64)

        splittedClusters_tc['Rz'] = np.sqrt(splittedClusters_tc.tc_x*splittedClusters_tc.tc_x + splittedClusters_tc.tc_y*splittedClusters_tc.tc_y)  / abs(splittedClusters_tc.tc_z)
        splittedClusters_tc = splittedClusters_tc[ (splittedClusters_tc['Rz'] < max_rz) & (splittedClusters_tc['Rz'] > min_rz) ]
        splittedClusters_tc = splittedClusters_tc.reset_index()
        splittedClusters_tc['Rz' + '_bin'] = pd.cut( splittedClusters_tc['Rz'], bins=kw['RzBinEdges'], labels=False )
        splittedClusters_tc['tc_phi' + '_bin'] = pd.cut( splittedClusters_tc['tc_phi'], bins=kw['PhiBinEdges'], labels=False )

        #convert bin ids back to values (central values in the bin)
        splittedClusters_tc['Rz'] = binConv(splittedClusters_tc['Rz' + '_bin'], binDistRz, min_rz)
        splittedClusters_tc['tc_phi'] = binConv(splittedClusters_tc['tc_phi' + '_bin'], binDistPhi, -np.pi)
        splittedClusters_tc.drop(['Rz' + '_bin', 'tc_phi' + '_bin'], axis=1, inplace=True)

        simAlgoPlots[fe] = (splittedClusters_3d, splittedClusters_tc)


    output_file( os.path.join('out', 'energyResolution.html') )
    _ncols = 1 if len(enresgrid)==1 else 2
    #show( gridplot(enresgrid, ncols=_ncols) )

    #########################################################################
    ################### PLOTTING: TRIGGER CELLS #############################
    #########################################################################
    tc_backgrounds = []
    for idx,grp in enumerate(groups):
        source = ColumnDataSource(grp)

        if log_scale:
            mapper = LogColorMapper(palette=mypalette,
                                    low=grp['nhits'].min(), high=grp['nhits'].max())
        else:
            mapper = LinearColorMapper(palette=mypalette,
                                       low=grp['nhits'].min(), high=grp['nhits'].max())

        title = title_common + '; {}'.format(tcSelections[idx])
        p = figure(width=1800, height=600, title=title,
                   x_range=Range1d(tcData['phi_center'].min()-SHIFTH, tcData['phi_center'].max()+SHIFTH),
                   y_range=Range1d(tcData['Rz_center'].min()-SHIFTV, tcData['Rz_center'].max().max()+SHIFTV),
                   tools="hover,box_select,box_zoom,reset,save", x_axis_location='below',
                   x_axis_type='linear', y_axis_type='linear',
                   )
        p.toolbar.logo = None

        p.rect( x='phi_center', y='Rz_center',
                source=source,
                width=binDistPhi, height=binDistRz,
                width_units='data', height_units='data',
                line_color='black', fill_color=transform('nhits', mapper)
               )

        color_bar = ColorBar(color_mapper=mapper,
                             ticker= ( LogTicker(desired_num_ticks=len(mypalette))
                                       if log_scale else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) ),
                             formatter=PrintfTickFormatter(format="%d")
                             )
        p.add_layout(color_bar, 'right')

        set_figure_props(p, phiBinCenters, rzBinCenters)

        p.hover.tooltips = [
            ("#hits", "@{nhits}"),
            ("min(eta)", "@{min_eta}"),
            ("max(eta)", "@{max_eta}"),
        ]

        tc_backgrounds.append( p )

    #########################################################################
    ################### PLOTTING: SIMULATION ################################
    #########################################################################
    tabs, pics = ([] for _ in range(2))

    for i,(_k,(df_3d,df_tc)) in enumerate(simAlgoPlots.items()):
        for ev in df_tc['event'].unique():
            ev_tc = df_tc[ df_tc.event == ev ]
            ev_3d = df_3d[ df_3d.event == ev ]
            _simCols_tc = [ 'tc_mipPt', 'tc_z', 'Rz',
                            'tc_phi', 'tc_eta', 'tc_id',
                            'genpart_exeta', 'genpart_exphi' ]
            ev_tc = ev_tc.filter(items=_simCols_tc)

            ev_3d['cl3d_Roverz'] = calculateRoverZfromEta(ev_3d.cl3d_eta)
            ev_3d['gen_Roverz'] = calculateRoverZfromEta(ev_3d.genpart_exeta)

            cl3d_pos_rz, cl3d_pos_phi = ev_3d['cl3d_Roverz'].unique(), ev_3d['cl3d_phi'].unique()
            gen_pos_rz, gen_pos_phi = ev_3d['gen_Roverz'].unique(), ev_3d['genpart_exphi'].unique()
            ev_3d = ev_3d.drop(['cl3d_Roverz', 'cl3d_eta', 'cl3d_phi'], axis=1, inplace=True)
            assert( len(gen_pos_rz) == 1 and len(gen_pos_phi) == 1 )

            groupby = ev_tc.groupby(['Rz', 'tc_phi'], as_index=False)
            group = groupby.count()

            energy_sum = groupby.sum()['tc_mipPt']
            eta_mins = groupby.min()['tc_eta']
            eta_maxs = groupby.max()['tc_eta']

            group = group.rename(columns={'tc_z': 'nhits'}, errors='raise')
            group.insert(0, 'min_eta', eta_mins)
            group.insert(0, 'max_eta', eta_maxs)
            group.insert(0, 'sum_en', energy_sum)

            source = ColumnDataSource(group)

            title = title_common + '; Algo: {}'.format(_k)
            p = figure(width=1100, height=300, title=title,
                       x_range=Range1d(kw['PhiBinEdges'][0]-SHIFTH, kw['PhiBinEdges'][-1]+SHIFTH),
                       y_range=Range1d(kw['RzBinEdges'][0]-SHIFTV, kw['RzBinEdges'][-1]+SHIFTV),
                       tools="hover,box_select,box_zoom,reset,save", x_axis_location='below',
                       x_axis_type='linear', y_axis_type='linear',
                       )

            mapoptions = dict( low=group['sum_en'].min(),
                               high=group['sum_en'].max() )
            if log_scale:
                mapper = LogColorMapper(palette=mypalette, **mapoptions)
            else:
                mapper = LinearColorMapper(palette=mypalette, **mapoptions)

            color_bar = ColorBar(color_mapper=mapper,
                                 ticker= ( LogTicker(desired_num_ticks=len(mypalette))
                                           if log_scale else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) ),
                                 formatter=PrintfTickFormatter(format="%d")
                                 )
            p.add_layout(color_bar, 'right')

            p.rect( x='tc_phi', y='Rz',
                    source=source,
                    width=binDistPhi, height=binDistRz,
                    width_units='data', height_units='data', line_color='black',
                    fill_color=transform('sum_en', mapper)
                   )

            cross_options = dict(size=25, angle=3.14159/4, line_width=4)
            p.cross(x=gen_pos_phi, y=gen_pos_rz, color='orange',
                    legend_label='Generated particle position', **cross_options)
            p.cross(x=cl3d_pos_phi, y=cl3d_pos_rz, color='red',
                    legend_label='3D cluster positions', **cross_options)

            set_figure_props(p, phiBinCenters, rzBinCenters)

            for bkg in tc_backgrounds:
                bkg.cross(x=gen_pos_phi, y=gen_pos_rz, color='orange', **cross_options)
                bkg.cross(x=cl3d_pos_phi, y=cl3d_pos_rz, color='red', **cross_options)

            p.hover.tooltips = [
                ("#hits", "@{nhits}"),
                ("#sum_en", "@{sum_en}"),
            ]

            pics.append( (p,ev) )
            tabs.append( Panel(child=p, title='{}'.format(ev)) )


    outname = os.path.join('out', 'triggerCellsOccup')
    output_file(outname+'.html')

    tc_panels = []
    for i,bkg in enumerate(tc_backgrounds):
        tc_panels.append( Panel(child=bkg, title='Selection {}'.format(i)) )

    show( layout([[enresgrid[0], Tabs(tabs=tabs)], [Tabs(tabs=tc_panels)]]) )
    # for pic,ev in pics:
    #     export_png(pic, filename=outname+'_event{}.png'.format(ev))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trigger cells occupancy.')
    parser.add_argument('--ledges', help='layer edges (if -1 is added the full range is also included)', default=[0,28], nargs='+', type=int)
    parser.add_argument('--pos_endcap', help='Use only the positive endcap.',
                        default=True, type=bool)
    parser.add_argument('--hcal', help='Consider HCAL instead of default ECAL.', action='store_true')
    parser.add_argument('--min_rz', help='Minimum value of R/z, as defined in CMSSW.',
                        default=0.076, type=float)
    parser.add_argument('--max_rz', help='Maximum value of R/z, as defined in CMSSW.',
                        default=0.58, type=float)
    parser.add_argument('-l', '--log', help='use color log scale', action='store_true')

    FLAGS = parser.parse_args()

    # ERROR: standalone does not receive tc_map
    plot_trigger_cells_occupancy(tc_map, FLAGS.pos_endcap,
                                 FLAGS.min_rz, FLAGS.max_rz,
                                 FLAGS.ledges,
                                 FLAGS.log)

