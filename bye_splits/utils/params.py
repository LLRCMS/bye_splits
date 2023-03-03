# coding: utf-8

_all_ = []

import os
from pathlib import Path
import sys
import argparse
import re

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import yaml

def read_task_params(task):
    with open(CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)
        d = cfg["base"]
        d.update(cfg[task])
    return d

LocalStorage = os.path.join(parent_dir, "data/new_algos/")
EOSStorage = lambda u, dir: os.path.join("/eos/user", u[0], u, dir)
CfgPath = os.path.join(parent_dir, "config.yaml")
