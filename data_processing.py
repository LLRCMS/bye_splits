"""
Functions used for data processing.
"""
import numpy as np

def preprocess(data, nbins_phi, nbins_rz, window_size):
    """Prepare the data to serve as input to the net in R/z slices"""
    # data variables' indexes
    assert data.attrs['columns'].tolist() == ['Rz', 'phi', 'Rz_bin', 'phi_bin']

    rz_idx = 0
    phi_idx = 1
    rzbin_idx = 2
    phibin_idx = 3

    data = data[()] #eager (lazy is the default)

    def _drop_columns(data, data_with_boundaries, idxs):
        """Drops the columns specified by indexes `idxs`, overriding data arrays."""
        drop = lambda d,obj: np.delete(d, obj=obj, axis=1)
        data = [drop(x, idxs) for x in data]
        data_with_boundaries = [drop(x, idxs) for x in data_with_boundaries]
        return data, data_with_boundaries

    def _split(data, split_index, sort_index, nbins_rz):
        """
        Creates a list of R/z slices, each ordered by phi.
        """
        data = data.astype('float32')

        # data sanity check
        rz_slices = np.unique(data[:,split_index])
        assert len(rz_slices) <= nbins_rz
        assert rz_slices.tolist() == [x for x in range(len(rz_slices))]

        # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
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

    def _normalize(data, index):
        """
        Standard max-min normalization of column `index`.
        """
        ref = data[:,index]
        ref = (ref-ref.min()) / (ref.max()-ref.min())
        return data

    def _set_boundary_conditions(data, window_size, phibin_idx, nbins_phi):
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

    data = _normalize(
        data,
        index=phi_idx
    )
    data = _split(
        data,
        split_index=rzbin_idx,
        sort_index=phibin_idx,
        nbins_rz=nbins_rz
    )
    data, data_with_boundaries, boundary_sizes = _set_boundary_conditions(
        data,
        window_size,
        phibin_idx,
        nbins_phi
    )
    data, data_with_boundaries = _drop_columns(
        data,
        data_with_boundaries,
        idxs=[rz_idx, rzbin_idx]
    )

    return data, data_with_boundaries, boundary_sizes
