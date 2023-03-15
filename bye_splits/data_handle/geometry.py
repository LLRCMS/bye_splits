# coding: utf-8

_all_ = [ 'GeometryData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
import yaml
import awkward as ak
import uproot as up
import pandas as pd
import logging

from utils import params, common
from data_handle.base import BaseData

class GeometryData(BaseData):
    """
    Handles V11 HGCAL geometry.
    Uses the orientation #1 of bottom row of slide 4 in
    https://indico.cern.ch/event/1111846/contributions/4675223/attachments/2372915/4052852/PartialsRotation.pdf
    """
    def __init__(self, reprocess=False, logger=None, is_tc=True):        
        super().__init__('', 'geom', reprocess, logger, is_tc)
        self.indata.path = self.cfg['geometry']['file']
        self.indata.adir = self.cfg['geometry']['dir']
        self.indata.tree = self.cfg['geometry']['tree']

        self.dataset = None
        self.dname = 'tc'
        self.var = common.dot_dict(self.cfg['varGeometry'])

        self.readvars = self._readvars()
        self.readvars.remove(self.var.wvs)
        self.readvars.remove(self.var.c)

        self.cu, self.cv = 'triggercellu', 'triggercellv'
        self.ceta, self.cphi = 'triggercellieta', 'triggercelliphi'
        self.wu, self.wv = 'waferu', 'waferv'

        ## geometry-specific parameters
        self.waferWidth = self.cfg['geometry']['waferSize'] #this defines all other sizes
        self.sensorSeparation = self.cfg['geometry']['sensorSeparation']
        self.N = 4 #number of cells per wafer side
        self.c30 = np.sqrt(3)/2 #cossine of 30 degrees
        self.t30 = 1/np.sqrt(3) #tangent of 30 degrees
        self.R = self.waferWidth / (3 * self.N)
        self.r = self.R * self.c30
        self.cellDistX = self.waferWidth/8.
        self.cellDistY = self.cellDistX * self.t30

    def cell_location(self, df):
        """
        Filter TC location in wafer based on the orientation: up-right, up-left and bottom.
        In V11 "orientation 0" actually corresponds to 6, as the concept of orientation is only introduced for later versions.
        """
        conds = {'UL': ~(((df.waferorient==6) & (df[self.cu] >= self.N) & (df[self.cu] > df[self.cv])) |
                         ((df.waferorient<6) & (df[self.cv] < self.N) & (df[self.cu] > df[self.cv]))),

                 'UR': ~(((df.waferorient==6) & (df[self.cv] >= self.N) & (df[self.cu] <= df[self.cv])) |
                         ((df.waferorient<6) & (df[self.cv] >= self.N) & (df[self.cu] >= self.N))),

                 'B':  ~(((df.waferorient==6) & (df[self.cu] < self.N) & (df[self.cv] < self.N)) |
                         ((df.waferorient<6) & (df[self.cu] < self.N) & (df[self.cv] >= df[self.cu])))}
        df['cloc'] = np.nan
        df['cloc'] = df.cloc.where(conds['B'], 'B')
        df['cloc'] = df.cloc.where(conds['UL'], 'UL')
        df['cloc'] = df.cloc.where(conds['UR'], 'UR')
        return df

    def cell_location_shift(self, df):
        """
        Shift the position of TCs according to their location to allow an aligned display.
        The true cell position becomes however more distant from the displayed cell center.
        """
        df.loc[:, 'tc_y'] = np.where(df['cloc']=='UR', df['tc_y']+self.cellDistY, df['tc_y'])
        df.loc[:, 'tc_x'] = np.where(df['cloc']=='B', df['tc_x']+self.cellDistX/2, df['tc_x'])
        df.loc[:, 'tc_y'] = np.where(df['cloc']=='B', df['tc_y']+self.cellDistY/2, df['tc_y'])
        return df

    def _display_cells(self, df, library):
        """Display silicon cells and scintillator tiles."""
        raise NotImplementedError()
    
    def _display_trigger_cells(self, df, library):
        """
        Display silicon trigger cells (groups of cells) and
        scintillator trigger tiles (groups of tiles).
        """
        df_si = df[(df.subdet==1) | (df.subdet==2)]
        df_si = self._display_silicon_tcs(df_si)

        df_sci = df[(df.subdet==10)]
        df_sci = self._display_scintillator_tcs(df_sci)

        return {'si': df_si, 'sci': df_sci}

    def _display_silicon_tcs(self, df):
        """Displays silicon trigger cells"""
        df['wafer_shift_x'] = (-2*df[self.wu] + df[self.wv])*(self.waferWidth+self.sensorSeparation)/2
        df['wafer_shift_y'] = (self.c30*df[self.wv])*(self.waferWidth+self.sensorSeparation)
        
        # cells_conversion = (lambda cu,cv: (1.5*(cv-cu)+0.5) * self.R,
        #                     lambda cu,cv: (cv+cu-2*self.N+1) * self.r) #orientation 6
        cells_conversion = (lambda cu,cv: (1.5*(cv-self.N)+1) * self.R,
                            lambda cu,cv: (2*cu-cv-self.N) * self.r) #orientation 7pu
  
        univ_wcenterx = cells_conversion[0](3,3) + self.cellDistX
        univ_wcentery = cells_conversion[1](3,3) + 3*self.cellDistY/2
        scale_x, scale_y = self.waferWidth/2, self.waferWidth/(2*self.c30)
        
        xcorners, ycorners = ([] for _ in range(2))
        xcorners.append(univ_wcenterx - scale_x)
        xcorners.append(univ_wcenterx)
        xcorners.append(univ_wcenterx + scale_x)
        xcorners.append(univ_wcenterx + scale_x)
        xcorners.append(univ_wcenterx)
        xcorners.append(univ_wcenterx - scale_x)
        ysub = np.sqrt(scale_y*scale_y-scale_x*scale_x)
        ycorners.append(univ_wcentery - ysub)
        ycorners.append(univ_wcentery - scale_y)
        ycorners.append(univ_wcentery - ysub)
        ycorners.append(univ_wcentery + ysub)
        ycorners.append(univ_wcentery + scale_y)
        ycorners.append(univ_wcentery + ysub)
            
        xpoint, x0, x1, x2, x3 = ({} for _ in range(5))
        ypoint, y0, y1, y2, y3 = ({} for _ in range(5))
        xaxis, yaxis = ({} for _ in range(2))
  
        df['tc_x_center'] = cells_conversion[0](df[self.cu],df[self.cv])
        df['tc_y_center'] = cells_conversion[1](df[self.cu],df[self.cv])
        df['tc_x'] = df.wafer_shift_x + df['tc_x_center']
        df['tc_y'] = df.wafer_shift_y + df['tc_y_center']
        df['wx_center'] = df.wafer_shift_x + univ_wcenterx # fourth vertex (center) for cu/cv=(3,3)
        df['wy_center'] = df.wafer_shift_y + univ_wcentery # fourth vertex (center) for cu/cv=(3,3)
  
        df = self.cell_location(df)
        # df = self.cell_location_shift(df)
        angle = 0 # must be changed for different wafer orientations
        for kloc in ('UL', 'UR', 'B'):
            cx_d, cy_d = df[df.cloc==kloc]['tc_x'], df[df.cloc==kloc]['tc_y']
            wc_x, wc_y = df[df.cloc==kloc]['wx_center'], df[df.cloc==kloc]['wy_center']
            # x0 refers to the x position the lefmost, down corner all diamonds (TCs)
            # x1, x2, x3 are defined in a counter clockwise fashion
            # same for y0, y1, y2 and y3
            # tc positions refer to the center of the diamonds
            if kloc == 'UL':
                x0.update({kloc: cx_d - self.cellDistX/2})
            elif kloc == 'UR':
                x0.update({kloc: cx_d - self.cellDistX/2})
            else:
                x0.update({kloc: cx_d - self.cellDistX})
  
            x1.update({kloc: x0[kloc][:] + self.cellDistX})
            if kloc in ('UL', 'UR'):
                x2.update({kloc: x1[kloc]})
                x3.update({kloc: x0[kloc]})
            else:
                x2.update({kloc: x1[kloc] + self.cellDistX})
                x3.update({kloc: x1[kloc]})
  
            if kloc == 'UL':
                y0.update({kloc: cy_d - (self.cellDistY/2 + self.cellDistX*self.t30)})
            elif kloc == 'UR':
                y0.update({kloc: cy_d - (self.cellDistY/2)})
            else:
                y0.update({kloc: cy_d})
  
            if kloc in ('UR', 'B'):
                y1.update({kloc: y0[kloc][:] - self.cellDistY})
            else:
                y1.update({kloc: y0[kloc][:] + self.cellDistY})
            if kloc in ('B'):
                y2.update({kloc: y0[kloc][:]})
            else:
                y2.update({kloc: y1[kloc][:] + 2*self.cellDistY})
            if kloc in ('UL', 'UR'):
                y3.update({kloc: y0[kloc][:] + 2*self.cellDistY})
            else:
                y3.update({kloc: y0[kloc][:] + self.cellDistY})
  
            x0[kloc], y0[kloc] = self.rotate(angle, x0[kloc], y0[kloc], wc_x, wc_y)
            x1[kloc], y1[kloc] = self.rotate(angle, x1[kloc], y1[kloc], wc_x, wc_y)
            x2[kloc], y2[kloc] = self.rotate(angle, x2[kloc], y2[kloc], wc_x, wc_y)
            x3[kloc], y3[kloc] = self.rotate(angle, x3[kloc], y3[kloc], wc_x, wc_y)
  
            keys = ['pos0','pos1','pos2','pos3']
            xaxis.update({
                kloc: pd.concat([x0[kloc],x1[kloc],x2[kloc],x3[kloc]],
                                axis=1, keys=keys)})
            yaxis.update(
                {kloc: pd.concat([y0[kloc],y1[kloc],y2[kloc],y3[kloc]],
                                 axis=1, keys=keys)})
            xaxis[kloc]['new'] = [[[[round(val, 3) for val in sublst]]]
                                  for sublst in xaxis[kloc].values.tolist()]
            yaxis[kloc]['new'] = [[[[round(val, 3) for val in sublst]]]
                                  for sublst in yaxis[kloc].values.tolist()]
            xaxis[kloc] = xaxis[kloc].drop(keys, axis=1)
            yaxis[kloc] = yaxis[kloc].drop(keys, axis=1)

        df['diamond_x'] = pd.concat(xaxis.values())
        df['diamond_y'] = pd.concat(yaxis.values())
  
        # define module corners' coordinates
        xcorners_str = ['corner1x','corner2x','corner3x','corner4x','corner5x','corner6x']
        assert len(xcorners_str) == len(xcorners)
        ycorners_str = ['corner1y','corner2y','corner3y','corner4y','corner5y','corner6y']
        assert len(ycorners_str) == len(ycorners)
        for i in range(len(xcorners)):
            df[xcorners_str[i]] = df.wafer_shift_x + xcorners[i]
        for i in range(len(ycorners)):
            df[ycorners_str[i]] = df.wafer_shift_y + ycorners[i]

        df['hex_x'] = df[xcorners_str].values.tolist()
        df['hex_x'] = df['hex_x'].map(lambda x: [[x]])
        df['hex_y'] = df[ycorners_str].values.tolist()
        df['hex_y'] = df['hex_y'].map(lambda x: [[x]])

        remove = ['tc_x_center', 'tc_y_center', 'wx_center', 'wy_center',
                  'cloc', 'wafer_shift_x', 'wafer_shift_y',
                  'triggercellieta', 'triggercelliphi', 'subdet']
        df = df.drop(xcorners_str + ycorners_str + remove, axis=1)
        return df

    def _display_scintillator_tcs(self, df):
        df = df.drop(['subdet', 'waferu', 'waferv', 'waferorient', 'waferpart',
                      'triggercellu', 'triggercellv', 'z',], axis=1)
        ops = ['max', 'min']
        minmax = df.groupby(['layer'], sort=False).agg({'triggercellieta': ops})
        minmax.columns = [x + '_ieta' for x in ops]
        df = df.merge(minmax, on='layer', how='inner')

        df['R'] = np.sqrt(df.x*df.x + df.y*df.y)
        df['phi'] = np.arctan2(df.y, df.x)

        # phi-adjacent trigger tiles have a phi separation of 2.5 degrees,
        # for all layers
        conv_to_radians = lambda k : k*np.pi/180.
        df['phimin'] = df.phi - conv_to_radians(1.25)
        df['phimax'] = df.phi + conv_to_radians(1.25)

        # assume the tiles are squared; their arclength (equal to the width
        # for small values) is equal to their height
        df['arc']  = df.R * conv_to_radians(2.5)
        arc_dist = df.arc/2
        df['rmin'] = df.R - arc_dist
        df['rmax'] = df.R + arc_dist

        # correct ieta boundaries (smaller trigger cells, 2 cells instead of 4)
        # to avoid physical overlap with adjacent ieta rings
        # corrections are layer dependent
        sel_lay_out = [38, 39, 50]
        df.loc[~(df.layer.isin(sel_lay_out)) &
               (df.triggercellieta==df.max_ieta), 'rmin'] += arc_dist/2
        df.loc[~(df.layer>=41) &
               (df.triggercellieta==df.min_ieta), 'rmax'] -= arc_dist/2
        
        df = df.drop(['R', 'arc', 'phi', 'min_ieta', 'max_ieta'], axis=1)
        return df
        
    def filter_columns(self, d):
        """Filter some columns to reduce memory usage"""
        cols_to_remove = ['color']
        cols = [x for x in d.fields if x not in cols_to_remove]
        return d[cols]

    def _from_parquet_to_geometry(self, ds, section, region):
        """
        Steps required for going from parquet format to full pandas geometry dataframe
        In principle all these steps could be done without using pandas,
        but the latter improves clarity.
        """
        ds = ak.to_dataframe(ds)
        ds = self.region_selection(ds, section, region)
        ds = self.prepare_for_display(ds)
        return ds

    def prepare_for_display(self, df, library='bokeh'):
        """Prepares dataframe to be displayed by certain libraries."""
        if self.is_tc:
            libraries = ('bokeh', )
            if library not in libraries:
                raise NotImplementedError()
            df = self._display_trigger_cells(df, library)
        else:
            libraries = ('bokeh', )
            if library not in libraries:
                raise NotImplementedError()
            #df = self._display_cells(df, library)
        return df

    def provide(self, section='si', region=None):
        """Provides a processed geometry dataframe to the client."""
        if not os.path.exists(self.outpath) or self.reprocess:
            if self.logger is not None:
                self.logger.debug('Storing geometry data...')
            self.store(section, region)

        if self.dataset is None: # use cached dataset (currently will never happen)
            if self.logger is not None:
                self.logger.debug('Retrieving geometry data...')
            ds = ak.from_parquet(self.outpath)
            self.dataset = self._from_parquet_to_geometry(ds, section, region)
        
        return self.dataset

    def _readvars(self):
        _vars = list(self.var.values())
        _no_tc_vars = ('triggercellu', 'triggercellv', 'waferorient', 'waferpart',
                       'waferu', 'waferv')
        if not self.is_tc:
            for k in _no_tc_vars:
                _vars.remove(k)
        return _vars
        
    def region_selection(self, df, section=None, region=None):
        """Select a specific geometry region. Used mostly for debugging."""
        if region is not None or (region is None and section is not None):
            regions = ('inside', 'periphery', 'wafer', None)
            assert region in regions

            if section == 'si':
                df = df[df.subdet==2]

            elif section == 'sci':
                df = df[df.subdet==10]
                
            if region == 'inside':
                if section == 'si':
                    df = df[((df[self.wu]==3) & (df[self.wv]==3)) |
                            ((df[self.wu]==3) & (df[self.wv]==4)) |
                            ((df[self.wu]==4) & (df[self.wv]==3)) |
                            ((df[self.wu]==4) & (df[self.wv]==4))]
                elif section == 'sci':
                    df = df[df[self.ceta]<4]
     
            elif region == 'periphery':
                if section == 'si':
                    df = df[((df[self.wu]==-6) & (df[self.wv]==3)) |
                            ((df[self.wu]==-6) & (df[self.wv]==4)) |
                            ((df[self.wu]==-7) & (df[self.wv]==3)) |
                            ((df[self.wu]==-8) & (df[self.wv]==2)) |
                            ((df[self.wu]==-8) & (df[self.wv]==1)) |
                            ((df[self.wu]==-7) & (df[self.wv]==2))]
                elif section == 'sci':
                    df = df[df[self.ceta]>16]
                    df = df[(df[self.cphi]>20) & (df[self.cphi]<40)]

            elif region == 'module':
                if section == 'si':
                    df = df[((df[self.wu]==3) & (df[self.wv]==3))]
                elif section == 'sci':
                    df = df[(df[self.ceta]==15) & (df[self.cphi]==45)]

        return df

    def rotate(self, angle, x, y, cx, cy):
        """Counter-clockwise rotation of 'angle' [radians]."""
        assert angle >= 0 and angle < 2 * np.pi
        ret_x = np.cos(angle)*(x-cx) - np.sin(angle)*(y-cy) + cx
        ret_y = np.sin(angle)*(x-cx) + np.cos(angle)*(y-cy) + cy
        return ret_x, ret_y

    def select(self):
        """Performs data selection for performance."""
        with up.open(self.indata.path) as f:
            tree = f[self.indata.tree_path]
            if self.logger is not None:
                self.logger.info(tree.show())
            data = tree.arrays(self.readvars)
            sel = data.zside==1
            fields = data.fields[:]

            for v in (self.var.side,):
                fields.remove(v)
            data = data[sel][fields]

            nl = int(self.cfg['geometry']['nlayersCEE'])
            subsel = (data.subdet==2) | (data.subdet==10)
            data['layer'] = data.layer + nl*ak.values_astype(subsel, to=int)
            data = data[((data.layer<=nl) & (data.layer%2==1)) | (data.layer>nl)]

            #below is correct but much slower (no operator isin in awkward)
            #this cut is anyways implemented in the skimmer
            #data = data[ak.Array([x not in params.disconnectedTriggerLayers for x in data.layer])]
            
            #data = data.drop_duplicates(subset=[self.var.cu, self.var.cv, self.var.l])
            # data[self.var.wv] = data.waferv
            # data[self.var.wvs] = -1 * data.waferv
            #data[self.var.c] = "#8a2be2"

        return data

    def store(self, section, region):
        """Stores the data selection in a parquet file for quicker access."""
        ds = self.select()
        if os.path.exists(self.outpath):
            os.remove(self.outpath)
        ds = self.filter_columns(ds)
        ak.to_parquet(ds, self.outpath)
        self.dataset = self._from_parquet_to_geometry(ds, section, region)
