import os
import numpy as np
import pandas as pd

def binConv(vals, dist, amin):
    """
    Converts bin indexes back to values (central values in the bin).
    Assumes equally-spaced bins.
    """
    return (vals*dist) + (dist/2) + amin

def calcRzFromEta(eta):
    """R/z = arctan(theta) [theta is obtained from pseudo-rapidity, eta]"""
    _theta = 2*np.arctan( np.exp(-1 * eta) )
    return np.arctan( _theta )

class SupressSettingWithCopyWarning:
    """
    Temporarily supress pandas SettingWithCopyWarning.
    It is known to ocasionally provide false positives.
    https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    """
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw

class dotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_column_idx(columns, col):
    return columns.index(col)

def get_detector_region_mask(df, region):
    """
    Obtain a mask to filter a specific detector region.
    subdetectors: - ECAL (1)
                  - HCAL silicon (2)
                  - HCAL scintillator (10)
    """
    if region == 'Si':
        subdetCond = (df.subdet == 1) | (df.subdet == 2)
    elif region == 'ECAL':
        subdetCond = (df.subdet == 1)
    elif region == 'MaxShower':
        subdetCond = ( (df.subdet == 1) &
                       (df.layer >= 8) & (df.layer <= 15) )

    df = df.drop(['subdet'], axis=1)
    return df, subdetCond

def get_html_name(script_name, extra=''):
    f = os.path.basename(script_name)
    f = f.split('.')
    f = f[0] + extra + '.html'
    return f

def tc_base_selection(df, region, pos_endcap, range_rz):
    if pos_endcap:
        df = df[ df.zside == 1 ] #only look at positive endcap
        df = df.drop(['zside'], axis=1)

    df['R'] = np.sqrt(df.x*df.x + df.y*df.y)
    df['Rz'] = df.R / abs(df.z)

    #the following cut removes almost no event at all
    df = df[ ((df['Rz'] < range_rz[1]) &
              (df['Rz'] > range_rz[0])) ]

    df, subdetCond = get_detector_region_mask(df, region)
    return df, subdetCond
