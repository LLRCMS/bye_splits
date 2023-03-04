# coding: utf-8

_all_ = [ 'EventDataParticle' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

import yaml

from utils import params
from data_handle.geometry import GeometryData
from data_handle.event import EventData

def get_data_reco_chain_start(nevents=500, reprocess=False):
    """Access event data."""
    data_part_opt = dict(tag='chain', reprocess=reprocess, debug=True)
    data_particle = EventDataParticle(particles='photons', **data_part_opt)
    ds_all, events = data_particle.provide_random_events(n=nevents, seed=42)
    # ds_all = data_particle.provide_events(events=[170004, 170015, 170017, 170014])

    tc_keep = {'event': 'event',
               'good_tc_waferu': 'tc_wu', 'good_tc_waferv': 'tc_wv',
               'good_tc_cellu': 'tc_cu', 'good_tc_cellv': 'tc_cv',
               'good_tc_layer': 'tc_layer',
               'good_tc_pt': 'tc_pt', 'good_tc_mipPt': 'tc_mipPt',
               'good_tc_x': 'tc_x', 'good_tc_y': 'tc_y', 'good_tc_z': 'tc_z',
               'good_tc_eta': 'tc_eta', 'good_tc_phi': 'tc_phi',
               'good_tc_cluster_id': 'tc_cluster_id'}

    ds_tc = ds_all['tc']
    ds_tc = ds_tc[tc_keep.keys()]
    ds_tc = ds_tc.rename(columns=tc_keep)

    gen_keep = {'event': 'event',
                'good_genpart_exeta': 'gen_eta', 'good_genpart_exphi': 'gen_phi', 
                'good_genpart_energy': 'gen_en'}
    ds_gen = ds_all['gen']
    ds_gen = ds_gen.rename(columns=gen_keep)

    cl_keep = {'event': 'event',
               'good_cl3d_eta': 'cl3d_eta', 'good_cl3d_phi': 'cl3d_phi',
               'good_cl3d_id': 'cl3d_id',
               'good_cl3d_energy': 'cl3d_en'}    
    ds_cl = ds_all['cl']
    ds_cl = ds_cl.rename(columns=cl_keep)

    return ds_gen, ds_cl, ds_tc

def EventDataParticle(particles, tag, reprocess, logger=None, debug=False):
    """Factory for EventData instances of different particle types"""
    if particles not in ('photons', 'electrons', 'pions'):
        raise ValueError('{} are not supported.'.format(particles))
        
    tag = particles + '_' + tag
    tag += '_debug' * debug
    
    with open(params.CfgPaths['data'], 'r') as afile:
        cfgdata = yaml.safe_load(afile)
        defevents = cfgdata['defaultEvents'][particles]

    with open(params.CfgPaths['prod'], 'r') as afile:
        cfgprod = yaml.safe_load(afile)
        path = cfgprod['io'][particles]
        
    return EventData(path, tag, defevents, reprocess, logger)
