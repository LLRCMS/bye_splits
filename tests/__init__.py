# coding: utf-8

__all__ = []

# adjust the path to import law
import os
import sys
parent_dir = os.path.abspath(__file__ + 2 * '/..')
sys.path.insert(0, parent_dir)

# import all tests
from .test_dummy import *
