"""
Plots TC distributions and particular events, before and after:
- smoothing
- trigger cell movement
The full TC distributions show the distributions *before* TC movement.

There is some code duplication with `_full/_f` and `_sel/_s` suffixes
to describe the full phase space and the one defined ("selected") by `pars["region"]`
"""
import os
import argparse
import numpy as np
import pandas as pd
import uproot as up
import random
import h5py
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
from utils.params import (
    fill_kwargs as kw,
    cluster_kwargs as cl_kw,
    smooth_kwargs as smooth_kw,
    seed_kwargs as seed_kw,
)
from utils.utils import (
    calcRzFromEta,
    fill_path,
    get_detector_region_mask,
    SupressSettingWithCopyWarning,
    tc_base_selection,
)

colors = ('orange', 'red', 'black')
def set_figure_props(p, hide_legend=True):
    """set figure properties"""
    p.axis.axis_line_color = 'black'
    p.axis.major_tick_line_color = 'black'
    p.axis.major_label_text_font_size = '10px'
    p.axis.major_label_standoff = 2
    p.xaxis.axis_label = r"$$\color{black} \phi$$"
    p.yaxis.axis_label = '$$R/z$$'
    if hide_legend:
        p.legend.click_policy='hide'
    
def plot_trigger_cells_occupancy(pars,
                                 plot_name,
                                 pos_endcap,
                                 layer_edges,
                                 nevents,
                                 log_scale=False,
								 show_html=False,
                                 **kw):
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

    # Inputs: Trigger Cells
    tcVariables = {'zside', 'subdet', 'layer', 'phi', 'eta', 'x', 'y', 'z', 'id'}
    assert(tcVariables.issubset(tcTree.keys()))
    tcVariables = list(tcVariables)
    tcData = tcTree.arrays(tcVariables, library='pd')

    # Inputs: Simulation 0 Pu Photons
    for fe,files in simAlgoFiles.items():
        name = fe
        dfs = []
        for afile in files:
            with pd.HDFStore(afile, mode='r') as store:
                dfs.append(store[name])
        simAlgoDFs[fe] = pd.concat(dfs)

    simAlgoNames = sorted(simAlgoDFs.keys())

    # Inputs: Cluster After Custom Iterative Algorithm
    outclusterplot = fill_path(cl_kw['ClusterOutPlot'], **pars)
    with pd.HDFStore(outclusterplot, mode='r') as store:
        splittedClusters_3d_local = store['data']

    tcData, subdetCond = tc_base_selection(tcData,
                                           pos_endcap=pos_endcap,
                                           region=pars['reg'],
                                           range_rz=(kw['MinROverZ'],
                                                     kw['MaxROverZ']))
    tcData.id = np.uint32(tcData.id)

    tcData_full = tcData[:]
    tcData_sel = tcData[subdetCond]

    copt = dict(labels=False)
    tcData_full['Rz_bin'] = pd.cut(tcData_full['Rz'],
                                   bins=kw['RzBinEdges'],
                                   **copt)
    tcData_full['phi_bin'] = pd.cut(tcData_full.phi,
                                    bins=kw['PhiBinEdges'],
                                    **copt)
    tcData_sel['Rz_bin'] = pd.cut(tcData_sel['Rz'],
                                  bins=kw['RzBinEdges'],
                                  **copt)
    tcData_sel['phi_bin'] = pd.cut(tcData_sel.phi,
                                   bins=kw['PhiBinEdges'],
                                   **copt)
    
    # Convert bin ids back to values (central values in each bin)
    tcData_full['Rz_center'] = binConv(tcData_full.Rz_bin, binDistRz, kw['MinROverZ'])
    tcData_full['phi_center'] = binConv(tcData_full.phi_bin, binDistPhi, kw['MinPhi'])
    tcData_sel['Rz_center'] = binConv(tcData_sel.Rz_bin, binDistRz, kw['MinROverZ'])
    tcData_sel['phi_center'] = binConv(tcData_sel.phi_bin, binDistPhi, kw['MinPhi'])
    _cols_drop = ['Rz_bin', 'phi_bin', 'Rz', 'phi']
    tcData_full = tcData_full.drop(_cols_drop, axis=1)
    tcData_sel = tcData_sel.drop(_cols_drop, axis=1)

    # if `-1` is included in layer_edges, the full selection is also drawn
    try:
        layer_edges.remove(-1)
        leftLayerEdges, rightLayerEdges = layer_edges[:-1], layer_edges[1:]
        leftLayerEdges.insert(0, 0)
        rightLayerEdges.insert(0, tcData_full.layer.max())
    except ValueError:
        leftLayerEdges, rightLayerEdges = layer_edges[:-1], layer_edges[1:]

    ledgeszip = tuple(zip(leftLayerEdges,rightLayerEdges))
    tcSelections = ['layer>{}, layer<={}'.format(x,y) for x,y in ledgeszip]
    grps_f, grps_s = ([] for _ in range(2))
    for lmin,lmax in ledgeszip:
        #full
        grps_f.append( tcData_full[ (tcData_full.layer>lmin) &
                                    (tcData_full.layer<=lmax) ] )
        groupby_full = grps_f[-1].groupby(['Rz_center', 'phi_center'],
                                          as_index=False)
        grps_f[-1] = groupby_full.count()
        eta_mins = groupby_full.min()['eta']
        eta_maxs = groupby_full.max()['eta']
        grps_f[-1].insert(0, 'min_eta', eta_mins)
        grps_f[-1].insert(0, 'max_eta', eta_maxs)
        grps_f[-1] = grps_f[-1].rename(columns={'z': 'ntc'})
        _cols_keep = ['phi_center', 'ntc', 'Rz_center',
                      'min_eta', 'max_eta']
        grps_f[-1] = grps_f[-1][_cols_keep]

        #sel
        grps_s.append( tcData_sel[ (tcData_sel.layer>lmin) &
                                    (tcData_sel.layer<=lmax) ] )
        groupby_sel = grps_s[-1].groupby(['Rz_center', 'phi_center'],
                                          as_index=False)
        grps_s[-1] = groupby_sel.count()
        eta_mins = groupby_sel.min()['eta']
        eta_maxs = groupby_sel.max()['eta']
        grps_s[-1].insert(0, 'min_eta', eta_mins)
        grps_s[-1].insert(0, 'max_eta', eta_maxs)
        grps_s[-1] = grps_s[-1].rename(columns={'z': 'ntc'})
        _cols_keep = ['phi_center', 'ntc', 'Rz_center',
                      'min_eta', 'max_eta']
        grps_s[-1] = grps_s[-1][_cols_keep]

    #########################################################################
    ################### DATA ANALYSIS: SIMULATION ###########################
    #########################################################################
    for i,fe in enumerate(kw['FesAlgos']):

        outfillplot = fill_path(kw['FillOutPlot'], **pars)
        with pd.HDFStore(outfillplot, mode='r') as store:
            splittedClusters_3d_cmssw = store[fe + '_3d']
            splittedClusters_tc = store[fe + '_tc']

        simAlgoPlots[fe] = (splittedClusters_3d_cmssw,
                            splittedClusters_tc,
                            splittedClusters_3d_local )

    #########################################################################
    ################### PLOTTING: TRIGGER CELLS #############################
    #########################################################################
    bckg_full, bckg_sel = ([] for _ in range(2))
    for idx,(grp_full,grp_sel) in enumerate(zip(grps_f,grps_s)):
        source_full = ColumnDataSource(grp_full)
        source_sel  = ColumnDataSource(grp_sel)

        mapper_class = LogColorMapper if log_scale else LinearColorMapper
        mapper = mapper_class(palette=mypalette,
                              low=grp_full['ntc'].min(),
                              high=grp_full['ntc'].max())

        title = title_common + '; {}'.format(tcSelections[idx])
        fig_opt = dict(title=title,
                       width=1800,
                       height=600,
                       x_range=Range1d(tcData_full.phi_center.min()-SHIFTH,
                                       tcData_full.phi_center.max()+SHIFTH),
                       y_range=Range1d(tcData_full.Rz_center.min()-SHIFTV,
                                       tcData_full.Rz_center.max().max()+SHIFTV),
                       tools="hover,box_select,box_zoom,reset,save",
                       x_axis_location='below',
                       x_axis_type='linear',
                       y_axis_type='linear')

        p_full = figure(**fig_opt)
        p_full.toolbar.logo = None
        p_sel  = figure(**fig_opt)
        p_sel.toolbar.logo = None

        rect_opt = dict(x='phi_center', y='Rz_center',
                        width=binDistPhi, height=binDistRz,
                        width_units='data', height_units='data',
                        line_color='black',
                        fill_color=transform('ntc', mapper))
        p_full.rect(source=source_full, **rect_opt)
        p_sel.rect(source=source_sel, **rect_opt)

        ticker = ( LogTicker(desired_num_ticks=len(mypalette))
                   if log_scale
                   else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) )
        color_bar = ColorBar(color_mapper=mapper,
                             title='#Hits',
                             ticker=ticker,
                             formatter=PrintfTickFormatter(format="%d"))
        p_full.add_layout(color_bar, 'right')
        p_sel.add_layout(color_bar, 'right')

        set_figure_props(p_full, hide_legend=False)
        set_figure_props(p_sel, hide_legend=False)        

        tooltips = [ ("#TriggerCells", "@{ntc}"),
                     ("min(eta)", "@{min_eta}"),
                     ("max(eta)", "@{max_eta}") ]
        p_full.hover.tooltips = tooltips
        p_sel.hover.tooltips = tooltips

        bckg_full.append( p_full )
        bckg_sel.append( p_sel )

    #########################################################################
    ################### PLOTTING: SIMULATION ################################
    #########################################################################
    assert len(kw['FesAlgos'])==1
    ev_panels = [] #pics = []

    for _k,(df_3d_cmssw,df_tc,df_3d_local) in simAlgoPlots.items():

        if  nevents > len(df_tc['event'].unique()):
            m = ( 'You are trying to plot more events ({}) than ' +
                  'those available in the dataset ({}).'
                  .format(len(df_tc['event'].unique()), nevents) )
            raise ValueError(m)
        
        event_sample = ( random.sample(df_tc['event'].unique()
                                       .astype('int').tolist(),
                                       nevents)
                        )
        for ev in event_sample:
            # Inputs: Energy 2D histogram after smoothing but before clustering
            outsmooth = fill_path(smooth_kw['SmoothOut'], **pars)
            with h5py.File(outsmooth, mode='r') as storeSmoothIn:

                kold = kw['FesAlgos'][0]+'_'+str(ev)+'_group_old'
                energies_post_smooth_old, _, _ = storeSmoothIn[kold]
                knew = kw['FesAlgos'][0]+'_'+str(ev)+'_group_new'
                energies_post_smooth_new, _, _ = storeSmoothIn[knew]

            # convert 2D numpy array to (rz_bin, phi_bin) pandas dataframe
            df_smooth_old = ( pd.DataFrame(energies_post_smooth_old)
                              .reset_index()
                              .rename(columns={'index': 'Rz_bin'}) )
            df_smooth_old = ( pd.melt(df_smooth_old,
                                      id_vars='Rz_bin',
                                      value_vars=[x for x in range(0,216)])
                             .rename(columns={'variable': 'phi_bin', 'value': 'energy_post_smooth_old'}) )
            df_smooth_old['Rz_center']  = binConv(df_smooth_old.Rz_bin,  binDistRz,  kw['MinROverZ'])
            df_smooth_old['phi_center'] = binConv(df_smooth_old.phi_bin, binDistPhi, kw['MinPhi'])
            
            df_smooth_new = ( pd.DataFrame(energies_post_smooth_new)
                            .reset_index()
                            .rename(columns={'index': 'Rz_bin'}) )
            df_smooth_new = pd.melt(df_smooth_new,
                                    id_vars='Rz_bin',
                                    value_vars=[x for x in range(0,216)]).rename(columns={'variable': 'phi_bin', 'value': 'energy_post_smooth_new'})

            # do not display empty (or almost empty) bins
            df_smooth_old = df_smooth_old[ df_smooth_old.energy_post_smooth_old > 0.1 ]
            df_smooth_new = df_smooth_new[ df_smooth_new.energy_post_smooth_new > 0.1 ]
            
            df_smooth_new['Rz_center']  = binConv(df_smooth_new.Rz_bin,  binDistRz,  kw['MinROverZ'])
            df_smooth_new['phi_center'] = binConv(df_smooth_new.phi_bin, binDistPhi, kw['MinPhi'])

            
            tools = "hover,box_select,box_zoom,reset,save"
            fig_opt = dict(width=900,
                           height=300,
                           x_range=Range1d(kw['PhiBinEdges'][0]-2*SHIFTH,
                                           kw['PhiBinEdges'][-1]+2*SHIFTH),
                           y_range=Range1d(kw['RzBinEdges'][0]-SHIFTV,
                                           kw['RzBinEdges'][-1]+SHIFTV),
                           tools=tools,
                           x_axis_location='below',
                           x_axis_type='linear',
                           y_axis_type='linear')

            ev_tc       = df_tc[ df_tc.event == ev ]
            ev_3d_cmssw = df_3d_cmssw[ df_3d_cmssw.event == ev ]
            ev_3d_local = df_3d_local[ df_3d_local.event == ev ]

            tc_cols = [ 'tc_mipPt', 'tc_z', 'Rz', 'tc_id',
                        'tc_eta', 'tc_eta_new', 
                        'phi_old', 'phi_new',
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
            ev_tc['phi_center_old'] = binConv(ev_tc.phi_bin_old, binDistPhi,
                                              kw['MinPhi'])
            ev_tc['phi_center_new'] = binConv(ev_tc.phi_bin_new, binDistPhi,
                                              kw['MinPhi'])
            _cols_drop = ['Rz_bin', 'phi_bin_old', 'phi_bin_new', 'Rz', 'phi_new']
            ev_tc = ev_tc.drop(_cols_drop, axis=1)

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

            groupby_old = ev_tc.groupby(['Rz_center',
                                         'phi_center_old'],
                                        as_index=False)
            groupby_new = ev_tc.groupby(['Rz_center',
                                         'phi_center_new'],
                                        as_index=False)
            group_old = groupby_old.count()
            group_new = groupby_new.count()

            _ensum = ( groupby_old.sum()['tc_mipPt'],
                       groupby_new.sum()['tc_mipPt'] )
            _etamins = ( groupby_old.min()['tc_eta'],
                         groupby_new.min()['tc_eta_new'] )
            _etamaxs = ( groupby_old.max()['tc_eta'],
                         groupby_new.max()['tc_eta_new'] )

            group_old = group_old.rename(columns={'tc_z': 'nhits'})
            group_old.insert(0, 'min_eta', _etamins[0])
            group_old.insert(0, 'max_eta', _etamaxs[0])
            group_old.insert(0, 'sum_en', _ensum[0])

            group_new = group_new.rename(columns={'tc_z': 'nhits'})
            group_new.insert(0, 'min_eta', _etamins[1])
            group_new.insert(0, 'max_eta', _etamaxs[1])
            group_new.insert(0, 'sum_en', _ensum[1])

            mapper_class = LogColorMapper if log_scale else LinearColorMapper
            ticker = ( LogTicker(desired_num_ticks=len(mypalette))
                      if log_scale
                      else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) )
            base_bar_opt = dict(ticker=ticker,
                                formatter=PrintfTickFormatter(format="%d"))

            rect_opt = dict( y='Rz_center',
                             width=binDistPhi, height=binDistRz,
                             width_units='data', height_units='data',
                             line_color='black' )

            seed_window = ( 'phi seeding step window: {}'
                           .format(seed_kw['WindowPhiDim']) )
            figs = []
            t_d = {0: ( 'Energy Density (before smoothing step, ' +
                        'before algo, {})'.format(seed_window) ),
                   1: ( 'Energy Density (before smoothing step, ' +
                        'after algo, {})'.format(seed_window) ),
                   2: ( 'Energy Density (after smoothing step, ' +
                        'before alg, {})'.format(seed_window) ),
                   3: ( 'Energy Density (after smoothing step, ' +
                        'after algo, {})'.format(seed_window) ),
                   4: ( 'Hit Density (before smoothing step, ' +
                        'before algo, {})'.format(seed_window) ),
                   5: ( 'Hit Density (before smoothing step, ' +
                        'after algo, {})'.format(seed_window) ) }
            group_d = {0: group_old,
                       1: group_new,
                       2: df_smooth_old,
                       3: df_smooth_new,
                       4: group_old,
                       5: group_new }
            hvar_d = {0: 'sum_en',
                      1: 'sum_en',
                      2: 'energy_post_smooth_old',
                      3: 'energy_post_smooth_new',
                      4: 'nhits',
                      5: 'nhits' }
            bvar_d = {0: 'Energy [mipPt]',
                      1: 'Energy [mipPt]',
                      2: 'Energy [mipPt]',
                      3: 'Energy [mipPt]',
                      4: '#Hits',
                      5: '#Hits' }
            toolvar_d = {0: ("EnSum", "@{sum_en}"),
                         1: ("EnSum", "@{sum_en}"),
                         2: ("EnSum", "@{energy_post_smooth_old}"),
                         3: ("EnSum", "@{energy_post_smooth_new}"),                         
                         4: ("#hits", "@{nhits}"),
                         5: ("#hits", "@{nhits}") }
            rec_opt_d = {0: dict(x='phi_center_old',
                                 source=ColumnDataSource(group_old),
                                 **rect_opt),
                         1: dict(x='phi_center_new',
                                 source=ColumnDataSource(group_new),
                                 **rect_opt),
                         2: dict(x='phi_center',
                                 source=ColumnDataSource(df_smooth_old),
                                 **rect_opt),
                         3: dict(x='phi_center',
                                 source=ColumnDataSource(df_smooth_new),
                                 **rect_opt),
                         4: dict(x='phi_center_old',
                                 source=ColumnDataSource(group_old),
                                 **rect_opt),
                         5: dict(x='phi_center_new',
                                 source=ColumnDataSource(group_new),
                                 **rect_opt) }

            base_cross_opt = dict(size=25, angle=np.pi/4, line_width=4)
            gen_label = 'Gen Particle Position'
            gen_cross_opt = dict(x=gen_pos_phi, y=gen_pos_rz,
                                 color=colors[0],
                                 legend_label=gen_label,
                                 **base_cross_opt)
            cmssw_label = 'CMSSW Cluster Position'
            cmssw_cross_opt = dict(x=cl3d_pos_phi, y=cl3d_pos_rz,
                                   color=colors[1],
                                   legend_label=cmssw_label,
                                   **base_cross_opt)
            local_label = 'Custom Cluster Position'
            local_cross_opt = dict(x=ev_3d_local.phinew,
                                   y=ev_3d_local.Rz,
                                   color=colors[2],
                                   legend_label=local_label,
                                    **base_cross_opt)

            for it in range(len(t_d.keys())):
                figs.append( figure(title=t_d[it], **fig_opt) )
                figs[-1].toolbar.logo = None

                map_opt = dict( low= group_d[it][ hvar_d[it] ].min(),
                                high=group_d[it][ hvar_d[it] ].max() )
                mapper = mapper_class(palette=mypalette, **map_opt)

                bar_opt = dict(title=bvar_d[it], **base_bar_opt)
                bar = ColorBar(color_mapper=mapper, **bar_opt)
                figs[-1].add_layout(bar, 'right')

                figs[-1].rect( fill_color=transform(hvar_d[it], mapper),
                              **rec_opt_d[it] )

                figs[-1].hover.tooltips = [ toolvar_d[it] ]
                figs[-1].cross(**gen_cross_opt)

                if it==1 or it==3 or it==5:
                    figs[-1].cross(**local_cross_opt)
                else:
                    figs[-1].cross(**cmssw_cross_opt)

                set_figure_props(figs[-1])

            cross1_opt = dict(x=gen_pos_phi, y=gen_pos_rz,
                              color=colors[0], **base_cross_opt)
            cross2_opt = dict(x=cl3d_pos_phi, y=cl3d_pos_rz,
                              color=colors[1], **base_cross_opt)
            for bkg1,bkg2 in zip(bckg_full,bckg_sel):
                bkg1.cross(**cross1_opt)
                bkg1.cross(**cross2_opt)
                bkg2.cross(**cross1_opt)
                bkg2.cross(**cross2_opt)

            #pics.append( (p,ev) )
            _lay = layout( [[figs[4], figs[5]], [figs[0],figs[1]], [figs[2],figs[3]]] )
            ev_panels.append( Panel(child=_lay,
                                    title='{}'.format(ev)) )

    output_file(plot_name)

    tc_panels_full, tc_panels_sel = ([] for _ in range(2))
    for i,(bkg1,bkg2) in enumerate(zip(bckg_full,bckg_sel)):
        tc_panels_full.append( Panel(child=bkg1,
                                     title='Full | Selection {}'.format(i)) )
        tc_panels_sel.append( Panel(child=bkg2,
                                    title='Region {} | Selection {}'.format(pars['reg'],i)) )

    lay = layout([[Tabs(tabs=ev_panels)],
                  [Tabs(tabs=tc_panels_sel)],
                  [Tabs(tabs=tc_panels_full)]])
    show(lay) if show_html else save(lay)
    # for pic,ev in pics:
    #     export_png(pic, filename=outname+'_event{}.png'.format(ev))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trigger cells occupancy.')
    parser.add_argument('--ledges', help='layer edges (if -1 is added the full range is also included)', default=[0,42], nargs='+', type=int)
    parser.add_argument('--pos_endcap', help='Use only the positive endcap.',
                        default=True, type=bool)
    parser.add_argument('--hcal', help='Consider HCAL instead of default ECAL.', action='store_true')
    parser.add_argument('-l', '--log', help='use color log scale', action='store_true')

    FLAGS = parser.parse_args()

    # ERROR: standalone does not receive tc_map
    plot_trigger_cells_occupancy(param,
                                 selection,
                                 FLAGS.pos_endcap,
                                 FLAGS.ledges,
                                 FLAGS.log)

