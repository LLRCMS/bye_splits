# coding: utf-8

_all_ = []

import os
import sys
import argparse

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

def add_parameters(parser):
    parser.add_argument(
        "--sel",
        default="all",
        type=str,
        help="Selection used to select cluster under study.",
    )
    parser.add_argument(
        "--reg",
        choices=("Si", "ECAL", "HCAL", "All", "MaxShower", "ExcludeMaxShower"),
        default="Si",
        type=str,
        help="Z region in the detector for the trigger cell geometry.",
    )
    seed_help = ' '.join((
        "Size of the window used for seeding.",
        "The size refers either to the phi direction (in R/z is always 1)",
        "or to cell u and v coordinates. A larger size",
        "captures more information but consumes more ",
        "firmware resources."
    ))
    parser.add_argument("--seed_window", help=seed_help, default=1, type=int)
    parser.add_argument(
        "--smooth_kernel",
        choices=("default", "flat_top"),
        default="default",
        type=str,
        help="Type of smoothing kernel being applied.",
    )
    parser.add_argument(
        "--cluster_algo",
        choices=("max_energy", "min_distance"),
        default="min_distance",
        type=str,
        help="Clustering algorithm applied.",
    )
    parser.add_argument(
        "--nevents",
        default=100,
        type=int,
        help="Number of events.",
    )
    parser.add_argument("--user", type=str, help="lxplus username.")
    parser.add_argument(
        "--cluster_studies",
        type=bool,
        help="Read+write files for cluster size studies.",
        default=False,
    )


def common_arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--no_seed',          action='store_true', help='do not run the seeding step')
    parser.add_argument('--no_cluster',       action='store_true', help='do not run the clustering step')
    parser.add_argument('--no_valid',         action='store_true', help='do not run any validation')
    parser.add_argument('--no_valid_seed',    action='store_true', help='do not run CS seed validation')
    parser.add_argument('--no_valid_cluster', action='store_true', help='do not run CS cluster validation')
    parser.add_argument('--sel',              default='all', type=str, help='Selection used to select cluster under study')
    parser.add_argument('--nevents',          default=100, type=int, help='Number of events')
    parser.add_argument('--smooth_kernel',    choices=('default', 'flat_top'), default='default', type=str, help='Type of smoothing kernel being applied')
    parser.add_argument('--seed_window',      default=1, type=int, help='seed_help')
    parser.add_argument('--cluster_algo',     choices=('max_energy', 'min_distance'), default='min_distance', type=str, help='Clustering algorithm applied.')
    parser.add_argument('--user',             default='bfontana', type=str, help='User selection')
    parser.add_argument('--reg',              choices=('Si', 'ECAL', 'HCAL', 'All', 'MaxShower', 'ExcludeMaxShower'), default='Si', type=str, help='Z region in the detector for the trigger cell geometry.')
    return parser

def parser_meta():
    parser = common_arguments('')
    parser.add_argument('--no_cs',     action='store_true', help='do not run the cs finder step')
    parser.add_argument("--no_fill",   action="store_true", help='do not run the filling step')
    parser.add_argument("--no_smooth", action="store_true", help='do not run the smoothing step')
    parsed_args = parser.parse_args()
    return parsed_args

def parser_new_chain():
    parser = common_arguments('Full reconstruction new chain')
    parser.add_argument('--no_cs', action='store_true', help='do not run the cs finder step')
    parsed_args = parser.parse_args()
    return parsed_args

def parser_default_chain():
    parser = common_arguments('Full reconstruction default chain')
    parser.add_argument("--no_fill",   action="store_true", help='do not run the filling step')
    parser.add_argument("--no_smooth", action="store_true", help='do not run the smoothing step')
    parsed_args = parser.parse_args()
    return parsed_args

def parser_radii_chain():
    parser = common_arguments('Full reconstruction default chain varing radii')
    parser.add_argument("--coefs",     action="store",      help="choose the radii",          default=[0.006,0.010,0.014,0.018,0.022,0.026,0.030])
    parser.add_argument("--particles", action="store",      help="choose the particles type", default='photons')
    parser.add_argument("--PU",        action="store",      help="choose the PU",             default='PU0')
    parser.add_argument("--event",     action="store",      help="choose the event",          default=None)
    parser.add_argument("--no_fill",   action="store_true", help='do not run the filling step')
    parser.add_argument("--no_smooth", action="store_true", help='do not run the smoothing step')
    parsed_args = parser.parse_args()
    return parsed_args

def parser_display_plotly():
    parser = common_arguments('Run the web application')
    parser.add_argument('--host', type=str,  default='llruicms01.in2p3.fr', help='host machine to run the application')
    parser.add_argument('--port', type=int,  default=8004,  help='port to use')
    parser.add_argument('--debug',type=bool, default=False, help='use debug mode to run the app')
    parser.add_argument("--no_fill",   action="store_true", help='do not run the filling step')
    parser.add_argument("--no_smooth", action="store_true", help='do not run the smoothing step')
    parsed_args = parser.parse_args()
    return parsed_args

