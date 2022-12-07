# coding: utf-8

_all_ = [ ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
from utils import common, params

class DataProcessing:
    def __init__(self, phi_bounds, bin_bounds):
        self.max_bound = 1.
        self.min_bound = 0.
        self.diff_bound = self.max_bound-self.min_bound
        self.shift_phi = np.pi #shift the phi range to [0; 2*Pi[
        self.phi_bounds = tuple( x + self.shift_phi for x in phi_bounds )

        #get parameters for the linear transformation (set between 0 and 1)
        self.a_norm_phi = self.diff_bound / (self.phi_bounds[1]-self.phi_bounds[0])
        self.b_norm_phi = self.max_bound - self.a_norm_phi * self.phi_bounds[1]

        self.bin_bounds = bin_bounds
        self.a_norm_bin = self.diff_bound / (self.bin_bounds[1] - self.bin_bounds[0])
        self.b_norm_bin = self.max_bound - self.a_norm_bin * self.bin_bounds[1]

    def preprocess( self, data,
                    nbins_phi,
                    nbins_rz,
                    window_size,
                    normalize=True ):
        """Prepare the data to serve as input to the net in R/z slices"""
        # data variables' indexes
        cols = ['R', 'Rz', 'phi', 'Rz_bin', 'phi_bin', 'id']
        assert data.attrs['columns'].tolist() == cols

        idx_d = common.dot_dict(dict(r      = 0,
                                     rz     = 1,
                                     phi    = 2,
                                     rzbin  = 3,
                                     phibin = 4,
                                     tc_id  = 5, ))

        data = data[()] #eager (lazy is the default)

        def _drop_columns_data(data, data_with_boundaries, idxs):
            """Drops the columns specified by indexes `idxs`, overriding data arrays."""
            drop = lambda d,obj: np.delete(d, obj=obj, axis=1)
            data = [drop(x, idxs) for x in data]
            data_with_boundaries = [drop(x, idxs) for x in data_with_boundaries]
            return data, data_with_boundaries

        def _shift_data(data, index):
            """
            Shift data. Originally meant to avoid negative values.
            """
            if index == idx_d['phi']:
                data[:,index] += self.shift_phi
            assert len(data[:,index][ data[:,index] < self.min_bound ]) == 0
            return data

        def _split_data(data, split_index, sort_index, nbins_rz):
            """
            Creates a list of R/z slices, each ordered by phi.
            """
            # data robustness checks
            rz_slices = np.unique(data[:,split_index])
            assert len(rz_slices) <= nbins_rz
            assert np.all(rz_slices[1:]-rz_slices[:-1] == 1.)

            # Https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
            # ordering is already done when the data is produced, the following line is not needed anymore
            # data = data[ data[:,split_index].argsort(kind='stable') ] # sort rows by Rz_bin "column"

            # https://stackoverflow.com/questions/31863083/python-split-numpy-array-based-on-values-in-the-array

            # `np.diff` catches all `data` indexes where the sorted bin changes
            data = np.split( data, np.where(np.diff(data[:,sort_index])<0)[0]+1 )

            assert len(data) == len(rz_slices)

            # data correct sorting check
            for elem in data:
                assert ( np.sort(elem[:,sort_index], kind='stable') == elem[:,sort_index] ).all()

            return data

        def _normalize_data(data, index):
            """
            Standard max-min normalization of column `index`.
            """
            if index == idx_d.phi:
                data[:,index] = self.a_norm_phi * data[:,index] + self.b_norm_phi
            else:
                raise ValueError()
            return data

        def _set_boundary_conditions_data(data, window_size, phibin_idx, nbins_phi):
            """
            Pad the original data to ensure boundary conditions over its Phi dimension.
            The boundary is done in terms of bins, not single trigger cells.
            `bound_cond_width` stands for the number of bins seen to the right and left.
            The right boundary is concatenated on the left side of `data`.
            """
            bound_cond_width = window_size - 1
            boundary_right_indexes = [ (x[:,phibin_idx] >= nbins_phi-bound_cond_width)
                                       for x in data ]
            boundary_right = [ x[y] for x,y in zip(data,boundary_right_indexes) ]
            boundary_sizes = [ len(x) for x in boundary_right ]
            data_with_boundaries = [ np.concatenate((br,x), axis=0)
                                     for br,x in zip(boundary_right,data) ]

            return data, data_with_boundaries, boundary_sizes

        if normalize:
            data = _shift_data(
                data,
                index=idx_d.phi
            )
            data = _normalize_data(
                data,
                index=idx_d.phi
            )
        data = _split_data(
            data,
            split_index=idx_d.rzbin,
            sort_index=idx_d.phibin,
            nbins_rz=nbins_rz
        )
        data, data_with_boundaries, boundary_sizes = _set_boundary_conditions_data(
            data,
            window_size,
            idx_d.phibin,
            nbins_phi
        )
        idxs_to_remove = [idx_d.rz, idx_d.rzbin]
        data, data_with_boundaries = _drop_columns_data(
            data,
            data_with_boundaries,
            idxs=idxs_to_remove
        )

        bins = []
        for rzslice in data:
            tmp = rzslice[:,idx_d.phi].astype(int)
            tmp = np.bincount(tmp, minlength=params.base_kw['NbinsPhi'])

            #normalization
            if normalize:
                tmp = self.a_norm_bin * tmp + self.b_norm_bin

            bins.append(tmp)

        # remove items from index dict
        for idx in idxs_to_remove:
            rem = [s for s in idx_d if idx_d[s]==idx]
            assert len(rem) == 1
            idx_d.pop(rem[0])

        # impose consecutive numbers
        c = 0
        for k in idx_d:
            idx_d[k] = c
            c += 1

        return data, bins, data_with_boundaries, boundary_sizes, idx_d

    def postprocess(self, data, bins):
        """Adapt the output of the neural network."""

        def _denormalize(data, bins):
            """Opposite of preprocess._normalize()."""
            new_data = (data - self.b_norm_phi) / self.a_norm_phi
            new_bins = (bins - self.b_norm_bin) / self.a_norm_bin
            return new_data, new_bins

        def _deshift(data, bins):
            """Opposite of preprocess._normalize()."""
            new_data = data - self.shift_phi
            new_bins = bins
            return new_data, new_bins

        new_data, new_bins = _denormalize(
            data,
            bins,
        )
        new_data, new_bins = _deshift(
            new_data,
            new_bins,
        )

        return new_data, new_bins
