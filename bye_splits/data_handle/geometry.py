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
    def __init__(self, inname='', reprocess=False, logger=None):
        super().__init__(inname, 'geom', reprocess, logger)
        self.dataset = None
        self.dname = 'tc'
        with open(params.viz_kw['CfgEventPath'], 'r') as afile:
            cfg = yaml.safe_load(afile)
            self.var = common.dot_dict(cfg['varGeometry'])

        self.readvars = list(self.var.values())
        self.readvars.remove(self.var.wvs)
        self.readvars.remove(self.var.c)

    def filter_columns(self, d):
        """Filter some columns to reduce memory usage"""
        cols_to_remove = ['x', 'y', 'z', 'color']
        cols = [x for x in d.fields if x not in cols_to_remove]
        return d[cols]

    def region_selection(self, df, region=None):
        if region is not None:
            regions = ('inside', 'periphery')
            assert region in regions
            if region == 'inside':
                df = df[((df['waferu']==3) & (df['waferv']==3)) |
                        ((df['waferu']==3) & (df['waferv']==4)) |
                        ((df['waferu']==4) & (df['waferv']==3)) |
                        ((df['waferu']==4) & (df['waferv']==4))]

            elif region == 'periphery':
                df = df[((df['waferu']==-6) & (df['waferv']==3)) |
                        ((df['waferu']==-6) & (df['waferv']==4)) |
                        ((df['waferu']==-7) & (df['waferv']==3)) |
                        ((df['waferu']==-8) & (df['waferv']==2)) |
                        ((df['waferu']==-8) & (df['waferv']==1)) |
                        ((df['waferu']==-7) & (df['waferv']==2))
                        ]

        # df = df[df.layer<=9]
        # df = df[df.waferpart==0]
        return df

    def prepare_for_display(self, df, library='bokeh'):
        libraries = ('bokeh', )
        assert library in libraries
        
        c30, s30, t30 = np.sqrt(3)/2, 1/2, 1/np.sqrt(3)
        N = 4
        waferWidth = 1
        R = waferWidth / (3 * N)
        r = R * c30
        cellDistX = waferWidth/8.
        cellDistY = cellDistX * t30

        scu, scv = 'triggercellu', 'triggercellv'
        swu, swv = 'waferu', 'waferv'
        df.loc[:, 'wafer_shift_x'] = (-2*df[swu] + df[swv])*waferWidth/2
        df.loc[:, 'wafer_shift_y'] = c30*df[swv]
    
        cells_conversion = (lambda cu,cv: (1.5*(cv-cu)+0.5) * R,
                            lambda cu,cv: (cv+cu-2*N+1) * r) #orientation 6

        univ_wcenterx = (1.5*(3-3) + 0.5)*R + cellDistX
        univ_wcentery = (3 + 3 - 2*N + 1) * r + 3*cellDistY/2
        scale_x, scale_y = waferWidth/2, waferWidth/(2*c30)
        
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
    
        def masks_location(location, ax, ay):
            """Filter TC location in wafer: up-right, up-left and bottom.
            The x/y location depends on the wafer orientation."""
            ux = np.sort(ax.unique())
            uy = np.sort(ay.unique())
            if len(ux)==0 or len(uy)==0:
                return pd.Series(dtype=float)
        
            if len(ux) != 2*N: #full wafers
                m = 'Length unique X values vs expected for full wafers: {} vs {}\n'.format(len(ux), 2*N)
                m += 'Fix.'
                raise AssertionError(m)
            if len(uy) != 4*N-1: #full wafers
                m = 'Length unique Y values vs expected for full wafers: {} vs {}\n'.format(len(uy), 4*N-1)
                m += 'Fix.'
                raise AssertionError(m)

            b = (-1/12, 0.)
            fx, fy = 1/8, (1/8)*t30 #multiplicative factors: cells are evenly spaced
            eps = 0.02 #epsilon, create an interval around the true values
            cx = abs(round((ux[0]-b[0])/fx)) 
            cy = abs(round((uy[N-1]-b[1])/fy))
            # -0.216, -0.144, -0.072, -0.000 /// +0.072, -0.000, -.072, -0.144

            filt_UL = ((ax > b[0]-(cx-0)*fx-eps) & (ax < b[0]-(cx-0)*fx+eps) & (ay > b[1]-(cy-0)*fy-eps) |
                       (ax > b[0]-(cx-1)*fx-eps) & (ax < b[0]-(cx-1)*fx+eps) & (ay > b[1]-(cy-1)*fy-eps) |
                       (ax > b[0]-(cx-2)*fx-eps) & (ax < b[0]-(cx-2)*fx+eps) & (ay > b[1]-(cy-2)*fy-eps) |
                       (ax > b[0]-(cx-3)*fx-eps) & (ax < b[0]-(cx-3)*fx+eps) & (ay > b[1]-(cy-3)*fy-eps))
            
            filt_UR = ((ax > b[0]-(cx-4)*fx-eps) & (ax < b[0]-(cx-4)*fx+eps) & (ay > b[1]-(cy-4)*fy-eps) |
                       (ax > b[0]-(cx-5)*fx-eps) & (ax < b[0]-(cx-5)*fx+eps) & (ay > b[1]-(cy-3)*fy-eps) |
                       (ax > b[0]-(cx-6)*fx-eps) & (ax < b[0]-(cx-6)*fx+eps) & (ay > b[1]-(cy-2)*fy-eps) |
                    (ax > b[0]-(cx-7)*fx-eps) & (ax < b[0]-(cx-7)*fx+eps) & (ay > b[1]-(cy-1)*fy-eps))

            if location == 'UL':
                return filt_UL
            elif location == 'UR':
                return filt_UR
            else: #bottom
                return (~filt_UL & ~filt_UR)

        xpoint, x0, x1, x2, x3 = ({} for _ in range(5))
        ypoint, y0, y1, y2, y3 = ({} for _ in range(5))
        xaxis, yaxis = ({} for _ in range(2))

        df.loc[:, 'tc_x_center'] = (1.5*(df[scv]-df[scu])+0.5) * R #orientation 6
        df.loc[:, 'tc_y_center'] = (df[scv]+df[scu]-2*N+1) * r #orientation 6
        df.loc[:, 'tc_x'] = df.wafer_shift_x + df['tc_x_center']
        df.loc[:, 'tc_y'] = df.wafer_shift_y + df['tc_y_center']
        wcenter_x = df.wafer_shift_x + univ_wcenterx # fourth vertex (center) for cu/cv=(3,3)
        wcenter_y = df.wafer_shift_y + univ_wcentery # fourth vertex (center) for cu/cv=(3,3)
        
        for loc_key in ('UL', 'UR', 'B'):
            masks_loc = masks_location(loc_key, df['tc_x_center'], df['tc_y_center'])
            cx_d, cy_d = df['tc_x'][masks_loc], df['tc_y'][masks_loc]
            wc_x, wc_y = wcenter_x[masks_loc], wcenter_y[masks_loc]
            
            # x0 refers to the x position the lefmost, down corner all diamonds (TCs)
            # x1, x2, x3 are defined in a counter clockwise fashion
            # same for y0, y1, y2 and y3
            # tc positions refer to the center of the diamonds
            if loc_key == 'UL':
                x0.update({loc_key: cx_d})
            elif loc_key == 'UR':
                x0.update({loc_key: cx_d})
            else:
                x0.update({loc_key: cx_d - cellDistX})
                
            x1.update({loc_key: x0[loc_key][:] + cellDistX})
            if loc_key in ('UL', 'UR'):
                x2.update({loc_key: x1[loc_key]})
                x3.update({loc_key: x0[loc_key]})
            else:
                x2.update({loc_key: x1[loc_key] + cellDistX})
                x3.update({loc_key: x1[loc_key]})

            if loc_key == 'UL':
                y0.update({loc_key: cy_d})
            elif loc_key == 'UR':
                y0.update({loc_key: cy_d})
            else:
                y0.update({loc_key: cy_d + cellDistY})

            if loc_key in ('UR', 'B'):
                y1.update({loc_key: y0[loc_key][:] - cellDistY})
            else:
                y1.update({loc_key: y0[loc_key][:] + cellDistY})
            if loc_key in ('B'):
                y2.update({loc_key: y0[loc_key][:]})
            else:
                y2.update({loc_key: y1[loc_key][:] + 2*cellDistY})
            if loc_key in ('UL', 'UR'):
                y3.update({loc_key: y0[loc_key][:] + 2*cellDistY})
            else:
                y3.update({loc_key: y0[loc_key][:] + cellDistY})

            angle = 0#5*np.pi/3
            # orientation 1 of bottom row of slide 4 in
            # https://indico.cern.ch/event/1111846/contributions/4675223/attachments/2372915/4052852/PartialsRotation.pdf
            x0[loc_key], y0[loc_key] = self.rotate(angle, x0[loc_key], y0[loc_key], wc_x, wc_y)
            x1[loc_key], y1[loc_key] = self.rotate(angle, x1[loc_key], y1[loc_key], wc_x, wc_y)
            x2[loc_key], y2[loc_key] = self.rotate(angle, x2[loc_key], y2[loc_key], wc_x, wc_y)
            x3[loc_key], y3[loc_key] = self.rotate(angle, x3[loc_key], y3[loc_key], wc_x, wc_y)
            
            keys = ['pos0','pos1','pos2','pos3']
            xaxis.update({
                loc_key: pd.concat([x0[loc_key],x1[loc_key],x2[loc_key],x3[loc_key]],
                                   axis=1, keys=keys)})
            yaxis.update(
                {loc_key: pd.concat([y0[loc_key],y1[loc_key],y2[loc_key],y3[loc_key]],
                                    axis=1, keys=keys)})
            
            xaxis[loc_key]['new'] = [[[[round(val, 3) for val in sublst]]]
                                     for sublst in xaxis[loc_key].values.tolist()]
            yaxis[loc_key]['new'] = [[[[round(val, 3) for val in sublst]]]
                                     for sublst in yaxis[loc_key].values.tolist()]
            xaxis[loc_key] = xaxis[loc_key].drop(keys, axis=1)
            yaxis[loc_key] = yaxis[loc_key].drop(keys, axis=1)

        df.loc[:, 'diamond_x'] = pd.concat(xaxis.values())
        df.loc[:, 'diamond_y'] = pd.concat(yaxis.values())
            
        # define module corners' coordinates
        xcorners_str = ['corner1x','corner2x','corner3x','corner4x','corner5x','corner6x']
        assert len(xcorners_str) == len(xcorners)
        ycorners_str = ['corner1y','corner2y','corner3y','corner4y','corner5y','corner6y']
        assert len(ycorners_str) == len(ycorners)
        for i in range(len(xcorners)):
            df[xcorners_str[i]] = df.wafer_shift_x + xcorners[i]
        for i in range(len(ycorners)):
            df[ycorners_str[i]] = df.wafer_shift_y + ycorners[i]

        df.loc[:, 'hex_x'] = df[xcorners_str].values.tolist()
        df.loc[:, 'hex_x'] = df['hex_x'].map(lambda x: [[x]])
        df.loc[:, 'hex_y'] = df[ycorners_str].values.tolist()
        df.loc[:, 'hex_y'] = df['hex_y'].map(lambda x: [[x]])
        df = df.drop(xcorners_str + ycorners_str + ['tc_x_center', 'tc_y_center'], axis=1)
        return df

    def provide(self, region=None):
        if not os.path.exists(self.outpath) or self.reprocess:
            if self.logger is not None:
                self.logger.info('Storing geometry data...')
            self.store()

        if self.dataset is None: #use cached dataset
            if self.logger is not None:
                self.logger.info('Retrieving geometry data...')
            ds = ak.from_parquet(self.outpath)
            ds = self.filter_columns(ds)
            ds = ak.to_dataframe(ds)
            ds = self.region_selection(ds, region)
            self.dataset = self.prepare_for_display(ds)
        
        return self.dataset

    def rotate(self, angle, x, y, cx, cy):
        """Counter-clockwise rotation of 'angle' [radians]."""
        assert angle >= 0 and angle < 2 * np.pi
        ret_x = np.cos(angle)*(x-cx) - np.sin(angle)*(y-cy) + cx
        ret_y = np.sin(angle)*(x-cx) + np.cos(angle)*(y-cy) + cy
        return ret_x, ret_y

    def select(self):
        with up.open(self.inpath) as f:
            tree = f[ os.path.join('hgcaltriggergeomtester', 'TreeTriggerCells') ]
            if self.logger is not None:
                self.logger.debug(tree.show())
            data = tree.arrays(self.readvars)
            sel = (data.zside==1) & (data.subdet==1)
            fields = data.fields[:]

            for v in (self.var.side, self.var.subd):
                fields.remove(v)
            data = data[sel][fields]
            data = data[data.layer%2==1]
            #below is correct but much slower (no operator isin in awkward)
            #this cut is anyways implemented in the skimmer
            #data = data[ak.Array([x not in params.disconnectedTriggerLayers for x in data.layer])]
            
            #data = data.drop_duplicates(subset=[self.var.cu, self.var.cv, self.var.l])
            data[self.var.wv] = data.waferv
            data[self.var.wvs] = -1 * data.waferv
            data[self.var.c] = "#8a2be2"

        return data

    def store(self):
        ds = self.select()
        if os.path.exists(self.outpath):
            os.remove(self.outpath)
        ak.to_parquet(ds, self.outpath)
        self.dataset = ak.to_dataframe(ds)
