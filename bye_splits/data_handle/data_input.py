# coding: utf-8

_all_ = ['InputData']

import os
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

class InputData:
    """Storage class for input strings required to access ROOT files and trees."""

    def __init__(self):
        self._path = None
        self._adir = None
        self._tree = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ('InputData instance:\n' +
             'path = {}\n'.format(self._path) +
             'dir  = {}\n'.format(self._adir) +
             'tree = {}\n'.format(self._tree))
        return s

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path):
        self._path = path

    @property
    def adir(self):
        return self._adir

    @adir.setter
    def adir(self, adir):
        self._adir = adir

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        self._tree = tree

    @property
    def tree_path(self):
        return self._adir + '/' + self._tree
