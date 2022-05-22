import os
import argparse
import numpy as np
import pandas as pd
import uproot as up
from random import random
#from bokeh.io import export_png

from bokeh.io import output_file, show, save
from bokeh.layouts import layout, row
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
from airflow.airflow_dag import (
    filling_kwargs as kw,
    clustering_kwargs as cl_kw,
    fill_path,
)

from random_utils import (
    calcRzFromEta,
    SupressSettingWithCopyWarning,
)

colors = ('orange', 'red', 'black')
def set_figure_props(p, xbincenters, ybincenters, hide_legend=True):
    """set figure properties"""
    p.axis.axis_line_color = 'black'
    p.axis.major_tick_line_color = 'black'
    p.axis.major_label_text_font_size = '10px'
    p.axis.major_label_standoff = 2
    p.xaxis.axis_label = r"$$\color{black} \phi$$"
    p.yaxis.axis_label = '$$R/z$$'
    if hide_legend:
        p.legend.click_policy='hide'
    
def plot_trigger_cells_occupancy(param,
                                 selection,
                                 trigger_cell_map,
                                 plot_name,
                                 pos_endcap,
                                 layer_edges,
                                 nevents,
                                 log_scale=False,
								 show_html=False,
                                 **kw):
    rzBinCenters = ['{:.2f}'.format(x) for x in ( kw['RzBinEdges'][1:] + kw['RzBinEdges'][:-1] ) / 2 ]
    phiBinCenters = ['{:.2f}'.format(x) for x in ( kw['PhiBinEdges'][1:] + kw['PhiBinEdges'][:-1] ) / 2 ]

    #assumes the binning is regular
    binDistRz = kw['RzBinEdges'][1] - kw['RzBinEdges'][0] 
    binDistPhi = kw['PhiBinEdges'][1] - kw['PhiBinEdges'][0]
    binConv = lambda vals,dist,amin : (vals*dist) + (dist/2) + amin

    SHIFTH, SHIFTV = 3*binDistPhi, binDistRz

    tcDataPath = os.path.join(kw['BasePath'], 'test_triggergeom.root')
    tcFile = up.open(tcDataPath)

    tcFolder = 'hgcaltriggergeomtester'
    tcTreeName = 'TreeTriggerCells'
    tcTree = tcFile[ os.path.join(tcFolder, tcTreeName) ]

    simDataPath = os.path.join(os.environ['PWD'], 'data', 'gen_cl3d_tc.hdf5')
    simAlgoDFs, simAlgoFiles, simAlgoPlots = ({} for _ in range(3))

    for fe in kw['FesAlgos']:
        simAlgoFiles[fe] = [ os.path.join(simDataPath) ]

    title_common = r'{} vs {} bins'.format(kw['NbinsPhi'], kw['NbinsRz'])
    if pos_endcap:
        title_common += '; Positive end-cap only'
    title_common += '; Min(R/z)={} and Max(R/z)={}'.format(kw['MinROverZ'],
                                                           kw['MaxROverZ'])

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
    ######### INPUTS: CLUSTERING AFTER CUSTOM ITERATIVE ALGORITHM ###########
    #########################################################################
    outclusteringplot = fill_path(cl_kw['ClusteringOutPlot'], param)
    with pd.HDFStore(outclusteringplot, mode='r') as store:
        splittedClusters_3d_local = store['data']

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
    tcData = tcData[ ((tcData['Rz'] < kw['MaxROverZ']) &
                      (tcData['Rz'] > kw['MinROverZ'])) ]

    tcData.id = np.uint32(tcData.id)

    tcData = tcData.merge(trigger_cell_map, on='id', how='right').dropna()

    assert_diff = tcData.phi_old - tcData.phi
    assert not np.count_nonzero(assert_diff)

    copt = dict(labels=False)
    tcData['Rz_bin'] = pd.cut( tcData['Rz'], bins=kw['RzBinEdges'], **copt )

    # to check the effect of NOT applying the tc mapping
    # replace `phi_new` by `phi_old`
    tcData['phi_bin_old'] = pd.cut( tcData.phi_old, bins=kw['PhiBinEdges'], **copt )
    tcData['phi_bin_new'] = pd.cut( tcData.phi_new, bins=kw['PhiBinEdges'], **copt )
    
    # Convert bin ids back to values (central values in each bin)
    tcData['Rz_center'] = binConv(tcData.Rz_bin, binDistRz, kw['MinROverZ'])
    tcData['phi_center_old'] = binConv(tcData.phi_bin_old, binDistPhi, kw['MinPhi'])
    tcData['phi_center_new'] = binConv(tcData.phi_bin_new, binDistPhi, kw['MinPhi'])
    
    tcData = tcData.drop(['Rz_bin', 'phi_bin_old', 'phi_bin_new', 'Rz', 'phi'], axis=1)

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
        groupby = groups[-1].groupby(['Rz_center', 'phi_center_old'], as_index=False)
        groups[-1] = groupby.count()

        eta_mins = groupby.min()['eta']
        eta_maxs = groupby.max()['eta']
        groups[-1].insert(0, 'min_eta', eta_mins)
        groups[-1].insert(0, 'max_eta', eta_maxs)
        groups[-1] = groups[-1].rename(columns={'z': 'ntc'})
        groups[-1] = groups[-1][['phi_center_old', 'ntc', 'Rz_center', 'min_eta', 'max_eta']]

    #########################################################################
    ################### DATA ANALYSIS: SIMULATION ###########################
    #########################################################################
    for i,fe in enumerate(kw['FesAlgos']):

        outfillingplot = fill_path(kw['FillingOutPlot'], param)
        with pd.HDFStore(outfillingplot, mode='r') as store:
            splittedClusters_3d_cmssw = store[fe + '_3d']
            splittedClusters_tc = store[fe + '_tc']

        simAlgoPlots[fe] = (splittedClusters_3d_cmssw,
                            splittedClusters_tc,
                            splittedClusters_3d_local )

    #########################################################################
    ################### PLOTTING: TRIGGER CELLS #############################
    #########################################################################
    tc_backgrounds = []
    for idx,grp in enumerate(groups):
        source = ColumnDataSource(grp)

        mapper_class = LogColorMapper if log_scale else LinearColorMapper
        mapper = mapper_class(palette=mypalette,
                              low=grp['ntc'].min(),
                              high=grp['ntc'].max())

        title = title_common + '; {}'.format(tcSelections[idx])
        p = figure(title=title,
                   width=1800,
                   height=600,
                   x_range=Range1d(tcData['phi_center_old'].min()-SHIFTH,
                                   tcData['phi_center_old'].max()+SHIFTH),
                   y_range=Range1d(tcData['Rz_center'].min()-SHIFTV,
                                   tcData['Rz_center'].max().max()+SHIFTV),
                   tools="hover,box_select,box_zoom,reset,save",
                   x_axis_location='below',
                   x_axis_type='linear',
                   y_axis_type='linear')
        p.toolbar.logo = None

        p.rect( x='phi_center_old', y='Rz_center',
                source=source,
                width=binDistPhi, height=binDistRz,
                width_units='data', height_units='data',
                line_color='black', fill_color=transform('ntc', mapper)
               )

        ticker = ( LogTicker(desired_num_ticks=len(mypalette))
                   if log_scale
                   else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) )
        color_bar = ColorBar(color_mapper=mapper,
                             title='#Hits',
                             ticker=ticker,
                             formatter=PrintfTickFormatter(format="%d"))
        p.add_layout(color_bar, 'right')

        set_figure_props(p, phiBinCenters, rzBinCenters, hide_legend=False)

        p.hover.tooltips = [
            ("#TriggerCells", "@{ntc}"),
            ("min(eta)", "@{min_eta}"),
            ("max(eta)", "@{max_eta}"),
        ]

        tc_backgrounds.append( p )

    #########################################################################
    ################### PLOTTING: SIMULATION ################################
    #########################################################################
    ev_panels = [] #pics = []

    for _k,(df_3d_cmssw,df_tc,df_3d_local) in simAlgoPlots.items():

        event_sample = random.sample(df_tc['event'].unique().astype('int'),
                                     nevents)
        for ev in event_sample:
            fig_opt = dict(width=900,
                           height=300,
                           x_range=Range1d(kw['PhiBinEdges'][0]-2*SHIFTH,
                                           kw['PhiBinEdges'][-1]+2*SHIFTH),
                           y_range=Range1d(kw['RzBinEdges'][0]-SHIFTV,
                                           kw['RzBinEdges'][-1]+SHIFTV),
                           tools="hover,box_select,box_zoom,reset,save",
                           x_axis_location='below',
                           x_axis_type='linear',
                           y_axis_type='linear')
            p1 = figure(title='Energy Density', **fig_opt)
            p2 = figure(title='Hit Density', **fig_opt)
            p1.toolbar.logo = None
            p2.toolbar.logo = None

            ev_tc       = df_tc[ df_tc.event == ev ]
            ev_3d_cmssw = df_3d_cmssw[ df_3d_cmssw.event == ev ]
            ev_3d_local = df_3d_local[ df_3d_local.event == ev ]

            tc_cols = [ 'tc_mipPt', 'tc_z', 'Rz',
                        'tc_eta', 'tc_id',
                        'phi_old',
                        'phi_new',
                        'genpart_exeta', 'genpart_exphi' ]
            ev_tc = ev_tc.filter(items=tc_cols)

            copt = dict(labels=False)
            ev_tc['Rz_bin']  = pd.cut(ev_tc.Rz,
                                      bins=kw['RzBinEdges'], **copt )
            ev_tc['phi_bin_old'] = pd.cut(ev_tc.phi_old,
                                      bins=kw['PhiBinEdges'], **copt )
            ev_tc['phi_bin_new'] = pd.cut(ev_tc.phi_new,
                                      bins=kw['PhiBinEdges'], **copt )
    
            # Convert bin ids back to values (central values in each bin)
            ev_tc['Rz_center'] = binConv(ev_tc.Rz_bin, binDistRz, kw['MinROverZ'])
            ev_tc['phi_center_old'] = binConv(ev_tc.phi_bin_old, binDistPhi, kw['MinPhi'])
            ev_tc['phi_center_new'] = binConv(ev_tc.phi_bin_new, binDistPhi, kw['MinPhi'])
            ev_tc = ev_tc.drop(['Rz_bin', 'phi_bin_old', 'phi_bin_new', 'Rz', 'phi_new'], axis=1)

            with SupressSettingWithCopyWarning():
                ev_3d_cmssw['cl3d_Roverz']=calcRzFromEta(ev_3d_cmssw.cl3d_eta)
                ev_3d_cmssw['gen_Roverz']=calcRzFromEta(ev_3d_cmssw.genpart_exeta)

            cl3d_pos_rz  = ev_3d_cmssw['cl3d_Roverz'].unique()
            cl3d_pos_phi = ev_3d_cmssw['cl3d_phi'].unique()
            gen_pos_rz   = ev_3d_cmssw['gen_Roverz'].unique()
            gen_pos_phi  = ev_3d_cmssw['genpart_exphi'].unique()
            drop_cols = ['cl3d_Roverz', 'cl3d_eta', 'cl3d_phi']
            ev_3d_cmssw = ev_3d_cmssw.drop(drop_cols, axis=1)
            assert( len(gen_pos_rz) == 1 and len(gen_pos_phi) == 1 )

            groupby_old = ev_tc.groupby(['Rz_center', 'phi_center_old'], as_index=False)
            groupby_new = ev_tc.groupby(['Rz_center', 'phi_center_new'], as_index=False)
            group_old = groupby_old.count()

            energy_sum = groupby_old.sum()['tc_mipPt']
            eta_mins   = groupby_old.min()['tc_eta']
            eta_maxs   = groupby_old.max()['tc_eta']

            group_old = group_old.rename(columns={'tc_z': 'nhits'}, errors='raise')
            group_old.insert(0, 'min_eta', eta_mins)
            group_old.insert(0, 'max_eta', eta_maxs)
            group_old.insert(0, 'sum_en', energy_sum)

            map_opt1 = dict( low=group_old['sum_en'].min(),
                             high=group_old['sum_en'].max() )
            map_opt2 = dict( low=group_old['nhits'].min(),
                             high=group_old['nhits'].max() )
            mapper_class = LogColorMapper if log_scale else LinearColorMapper
            mapper1 = mapper_class(palette=mypalette, **map_opt1)
            mapper2 = mapper_class(palette=mypalette, **map_opt2)

            ticker = ( LogTicker(desired_num_ticks=len(mypalette))
                      if log_scale
                      else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) )
            base_bar_opt = dict(ticker=ticker,
                                formatter=PrintfTickFormatter(format="%d"))
            bar1_opt = dict(title='Energy [mipPt]',
                            **base_bar_opt)
            bar2_opt = dict(title='#Hits',
                            **base_bar_opt)
            bar1 = ColorBar(color_mapper=mapper1, **bar1_opt)
            bar2 = ColorBar(color_mapper=mapper2, **bar2_opt)
            p1.add_layout(bar1, 'right')
            p2.add_layout(bar2, 'right')

            source_old = ColumnDataSource(group_old)
            plot_opt = dict(x='phi_center_old', y='Rz_center',
                            source=source_old,
                            width=binDistPhi, height=binDistRz,
                            width_units='data', height_units='data',
                            line_color='black')
            p1.rect( fill_color=transform('sum_en', mapper1), **plot_opt )
            p2.rect( fill_color=transform('nhits',  mapper2), **plot_opt )

            base_cross_opt = dict(size=25, angle=np.pi/4, line_width=4)
            gen_cross_opt = dict(x=gen_pos_phi, y=gen_pos_rz,
                                 color=colors[0],
                                 legend_label='Gen Particle Position',
                                 **base_cross_opt)
            p1.cross(**gen_cross_opt)
            p2.cross(**gen_cross_opt)
            cmssw_cross_opt = dict(x=cl3d_pos_phi, y=cl3d_pos_rz,
                                   color=colors[1],
                                   legend_label='CMSSW Cluster Position',
                                   **base_cross_opt)
            p1.cross(**cmssw_cross_opt)
            p2.cross(**cmssw_cross_opt)
            local_cross_opt = dict(x=ev_3d_local.phi, y=ev_3d_local.Rz,
                                   color=colors[2],
                                   legend_label='Custom Cluster Position',
                                   **base_cross_opt)
            p1.cross(**local_cross_opt)
            p2.cross(**local_cross_opt)
            set_figure_props(p1, phiBinCenters, rzBinCenters)
            p1.hover.tooltips = [ ("EnSum", "@{sum_en}"),
                                  ("min(eta)", "@{min_eta}"),
                                  ("max(eta)", "@{max_eta}") ]

            set_figure_props(p2, phiBinCenters, rzBinCenters)
            p2.hover.tooltips = [ ("#hits", "@{nhits}"),
                                  ("min(eta)", "@{min_eta}"),
                                  ("max(eta)", "@{max_eta}") ]
    


            for bkg in tc_backgrounds:
                bkg.cross(x=gen_pos_phi, y=gen_pos_rz,
                          color=colors[0], **base_cross_opt)
                bkg.cross(x=cl3d_pos_phi, y=cl3d_pos_rz,
                          color=colors[1], **base_cross_opt)
                # bkg.cross(x=ev_3d_local.phi, y=ev_3d_local.Rz,
                #           color=colors[2], **base_cross_opt)

            #pics.append( (p,ev) )
            r = row(p1,p2)
            ev_panels.append( Panel(child=r, title='{}'.format(ev)) )

    output_file(plot_name)

    tc_panels = []
    for i,bkg in enumerate(tc_backgrounds):
        tc_panels.append( Panel(child=bkg, title='Selection {}'.format(i)) )

    #lay = layout([[enresgrid[0], Tabs(tabs=tabs)], [Tabs(tabs=tc_panels)]])
    lay = layout([[Tabs(tabs=ev_panels)], [Tabs(tabs=tc_panels)]])
    show(lay) if show_html else save(lay)
    # for pic,ev in pics:
    #     export_png(pic, filename=outname+'_event{}.png'.format(ev))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trigger cells occupancy.')
    parser.add_argument('--ledges', help='layer edges (if -1 is added the full range is also included)', default=[0,28], nargs='+', type=int)
    parser.add_argument('--pos_endcap', help='Use only the positive endcap.',
                        default=True, type=bool)
    parser.add_argument('--hcal', help='Consider HCAL instead of default ECAL.', action='store_true')
    parser.add_argument('-l', '--log', help='use color log scale', action='store_true')

    FLAGS = parser.parse_args()

    # ERROR: standalone does not receive tc_map
    plot_trigger_cells_occupancy(param,
                                 tc_map, FLAGS.pos_endcap,
                                 FLAGS.ledges,
                                 FLAGS.log)

