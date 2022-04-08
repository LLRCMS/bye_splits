import numpy as np

def binConv(vals, dist, amin):
    """
    Converts bin indexes back to values (central values in the bin).
    Assumes equally-spaced bins.
    """
    return (vals*dist) + (dist/2) + amin

def calculateRoverZfromEta(eta):
    """R/z = arctan(theta) [theta is obtained from pseudo-rapidity, eta]"""
    _theta = 2*np.arctan( np.exp(-1 * eta) )
    return np.arctan( _theta )

class dotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
