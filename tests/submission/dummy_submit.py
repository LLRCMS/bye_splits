# coding: utf-8

_all_ = ['dummy']

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from bye_splits.utils import common

import argparse


def dummy(pars):
    """Dummy script for testing HT Condor job submission.
    Prints passed argument key/vals and the type of <val>."""

    for key, val in pars.items():
        print("Passed {} for --{}, read as type {}.".format(val, key, type(val)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--float_arg", help="Dummy float", required=True, type=float)
    parser.add_argument("--str_arg", help="Dummy string.", required=True, type=str)
    parser.add_argument("--gen_arg", help="Dummy gen.")
    parser.add_argument("--store_arg", help="Dummy store_strue variable.", action="store_true")
    
    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    dummy(pars)