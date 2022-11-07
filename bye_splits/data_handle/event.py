# coding: utf-8

_all_ = [ 'EventData' ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

from utils import params

class EventData:
    def __init__(self, fname):
        self.path = (Path(__file__).parent.absolute().parent.parent /
                     params.DataFolder / self.fname )
