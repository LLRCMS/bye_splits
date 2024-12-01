# coding: utf-8

_all_ = ['dummy']

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from bye_splits.utils import common
from bye_splits.utils.job_helpers import Arguments

import argparse

arg_dict = {
    "--float_arg": {
        "help": "Dummy float",
        "type": float,
        "required": True
    },
    "--str_arg": {
        "help": "Dummy string.",
        "type": str,
        "required": True,
    },
    # Generic argument will be interpreted as a string
    "--gen_arg": {
        "help": "Dummy generic.",
        "required": False,
        "default": None
    },
    "--flag": {
        "help": "Dummy flag number 2.",
        "action": "store_true",
        "required": False
    }
}

def dummy(pars):
    """Dummy script for testing HT Condor job submission.
    Prints passed argument key/vals and the type of <val>."""

    for key, val in pars.items():
        print("Passed {} for --{}, read as type {}.".format(val, key, type(val)))


if __name__ == "__main__":
    args = Arguments(script=__file__)
    FLAGS = args.add_args(description="A dummy script to submit.", arg_dict=arg_dict)
    dummy(FLAGS)