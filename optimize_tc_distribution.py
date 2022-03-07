import h5py
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D

import sys
np.set_printoptions(threshold=sys.maxsize)

def tensorflow_assignment(tensor, mask, op):
    """
    Emulate assignment by creating a new tensor.
    The mask must be 1 where the assignment is intended, 0 everywhere else.
    """
    assert tensor.shape == mask.shape
    other = op(tensor)
    return tensor * mask + other * (1 - mask)

class DataProcessor():
    """Prepare the data to serve as input to the net in R/z slices"""
    def __init__(self, data, nbinsPhi, nbinsRz, window_size):
        # data variables' indexes
        assert data.attrs['columns'].tolist() == ['Rz', 'phi', 'Rz_bin', 'phi_bin']

        self.rz_idx = 0
        self.phi_idx = 1
        self.rzbin_idx = 2
        self.phibin_idx = 3

        self.data = data[()] #eager (lazy is the default)
        self.nbins = (nbinsPhi, nbinsRz)
        self.bound_cond_width = 3

        # data preprocessing
        self._normalize(index=self.phi_idx)
        self._split(sort_index=self.rzbin_idx)

        # add cyclic boundaries
        self.data_with_boundaries = self.set_boundary_conditions(window_size)

        # drop unneeded columns
        self._drop_columns(idxs=[self.rz_idx, self.rzbin_idx])

    def _drop_columns(self, idxs):
        """Drops the columns specified by indexes `idxs`, overriding data arrays."""
        drop = lambda d,obj: np.delete(d, obj=obj, axis=1)
        self.data = [drop(x, idxs) for x in self.data]
        self.data_with_boundaries = [drop(x, idxs) for x in self.data_with_boundaries]

    def _split(self, sort_index):
        """
        Creates a list of R/z slices, each ordered by phi.
        """
        self.data = self.data.astype('float32')

        rz_slices = np.unique(self.data[:,sort_index])
        assert len(rz_slices) <= self.nbins[1]
        assert rz_slices.tolist() == [x for x in range(len(rz_slices))]

        # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
        self.data = self.data[ self.data[:,sort_index].argsort() ] # sort rows by Rz_bin "column"
        # https://stackoverflow.com/questions/31863083/python-split-numpy-array-based-on-values-in-the-array

        # `np.diff` catches all `data` indexes where the sorted bin changes
        self.data = np.split( self.data, np.where(np.diff(self.data[:,sort_index]))[0]+1 )

    def _normalize(self, index):
        """
        Standard max-min normalization of column `index`.
        """
        ref = self.data[:,index]
        ref = (ref-ref.min()) / (ref.max()-ref.min())

    def set_boundary_conditions(self, window_size):
        """
        Pad the original data to ensure boundary conditions over its Phi dimension.
        The boundary is done in terms of bins, not single trigger cells.
        `self.bound_cond_width` stands for the number of bins seen to the right and left.
        The right boundary is concatenated on the left side of `data`.
        """
        bound_cond_width = window_size - 1
        boundary_right_indexes = [ (x[:,self.phibin_idx] >= self.nbins[0]-bound_cond_width)
                                   for x in self.data ]
        boundary_right = [ x[y] for x,y in zip(self.data,boundary_right_indexes) ]

        self.data_with_boundaries = [ np.concatenate((br,x), axis=0)
                                      for br,x in zip(boundary_right,self.data) ]
        return self.data_with_boundaries

# https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class
class Architecture(tf.keras.Model):
    """Neural network model definition."""
    def __init__(self, inshape, kernel_size):
        super().__init__()
        assert len(inshape)==1

        self.inshape = inshape

        self.conv1 = Conv1D( input_shape=(self.inshape[0],1),
                             filters=32,
                             kernel_size=kernel_size,
                             strides=1,
                             padding='same', #'valid' means no padding
                             activation='relu',
                             use_bias=True )
        self.conv2 = Conv1D( filters=8,
                             kernel_size=kernel_size,
                             strides=1,
                             padding='same', #'valid' means no padding
                             activation='relu',
                             use_bias=True )
        self.flatten = Flatten()
        self.dense = Dense( units=self.inshape[0],
                            activation='relu',
                            use_bias=True )

    def __call__(self, x):
        x = tf.reshape(x, shape=(-1, x.shape[0], 1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)

        return x

class TriggerCellDistributor(tf.Module):
    """Neural net workings"""
    def __init__(self, inshape, kernel_size, window_size,
                 phibounds, nbinsphi, rzbounds, nbinsrz,
                 pars=(1, 0.5, 1)):
        """
        Manages quantities related to the neural model being used.
        Args: - inshape: Shape of the input data
              - kernel_size: Length of convolutional kernels
              - window_size: Number of bins considered for each variance calculation.
              Note this is not the same as number of trigger cells (each bin has
              multiple trigger cells).
        """
        self.architecture = Architecture(inshape, kernel_size)
        self.kernel_size = kernel_size
        self.boundary_width = window_size-1
        self.phibounds, self.nbinsphi = phibounds, nbinsphi
        self.rzbounds, self.nbinsrz = rzbounds, nbinsrz

        assert len(pars)==3
        self.pars = pars

        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def calc_loss(self, indata, inbins, outdata):
        """
        Calculates the model's loss function. Receives slices in R/z as input.
        Each array value corresponds to a trigger cell.
        """
        opt = {'summarize': 10} #number of entries to print if assertion fails
        ass1 = tf.debugging.Assert(tf.math.reduce_min(inbins)==0, [inbins], **opt)
        ass2 = tf.debugging.Assert(tf.math.reduce_max(inbins)==self.nbinsphi-1, [inbins], **opt)

        # bin the output of the neural network
        outbins = tf.histogram_fixed_width_bins(outdata,
                                                value_range=self.phibounds,
                                                nbins=self.nbinsphi)
        ass3 = tf.debugging.Assert(tf.math.reduce_min(outbins)>=0, [outbins], **opt)
        ass4 = tf.debugging.Assert(tf.math.reduce_max(outbins)<=self.nbinsphi-1, [outbins], **opt)

        # convert the bin ids coming before 0 (shifted boundaries) to negative ones
        # if this looks convoluted, blame tensorflow, which still does not support tensor assignment
        # a new tensor must be created instead
        subtract_max = lambda x: x - tf.math.reduce_max(x)+1
        where_neg_diff = tf.where(inbins[1:]-inbins[:-1]<0)
        print('InBins: ', inbins.numpy())
        #print('Where: ', where_neg_diff)
        quit()

        print(   inbins[:where_neg_diff[0]] )

        inbins = tensorflow_assignment( tensor=inbins,
                                        mask=tf.concat((tf.ones(inbins[:tf.math.argmin(inbins)].shape[0]),
                                                        tf.zeros(inbins[tf.math.argmin(inbins):].shape[0])),
                                                       axis=0),
                                        op=subtract_max,
                                       )
        outbins = tensorflow_assignment( tensor=outbins,
                                         mask=tf.concat((tf.ones(outbins[:tf.math.argmin(outbins)].shape[0]),
                                                        tf.zeros(outbins[tf.math.argmin(outbins):].shape[0])),
                                                       axis=0),
                                         op=subtract_max,
                                       )

        # replicated boundaries should be the same
        ass5 = tf.debugging.Assert(
            condition=(indata[inbins<0] ==
                       indata[inbins>tf.math.reduce_max(inbins)-self.boundary_width]),
            data=[#indata[inbins<0],
                  indata[inbins>tf.math.reduce_max(inbins)-self.boundary_width]],
            **opt)
        
        with tf.control_dependencies([ass1,ass2,ass3,ass4,ass5]):
            # calculate the variance between adjacent bins
            variance_loss = 0
            for ibin in tf.unique(outbins)[:-self.boundary_width]:
                idxs = (outbins >= ibin) & (outbins <= ibin+self.boundary_width)
                variance_loss += tf.math.reduce_variance(indata[idxs])

            # calculate the earth-mover's distance between the net output and the original data
            wasserstein_loss = tf.cumsum(indata) - tf.cumsum(outdata)
            wasserstein_loss = tf.abs( wasserstein_loss )
            wasserstein_loss = tf.reduce_sum( wasserstein_loss )

            boundary_sanity_loss = tf.abs( outdata[outbins<0] -
                                           outdata[outbins>outbins.max()-self.boundary_width] )

            loss_pars = [ tf.Variable(x, trainable=False) for x in self.pars ]

        return ( loss_pars[0] * variance_loss +
                 loss_pars[1] * wasserstein_loss +
                 loss_pars[2] * boundary_sanity_loss )

    # @tf.function(
    #     input_signature=(
    #         tf.TensorSpec(shape=[], dtype=tf.float32),
    #         tf.TensorSpec(shape=[], dtype=tf.int32))
    # )
    #@tf.function
    def train(self, indata, inbins):
        # Reset the metrics at the start of the next epoch
        self.train_loss.reset_states()

        # for n in tf.range(steps): # any future loop within one epoch is done here
        with tf.GradientTape() as tape:
            tape.watch(indata)
            tape.watch(inbins)

            prediction = self.architecture(indata)
            loss = self.calc_loss(indata, inbins, prediction)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.train_loss(loss)
        return loss, prediction


def optimization(algo, **kw):
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    #store_out = h5py.File(kw['OptimizationOut'], mode='w')

    assert len(store_in.keys()) == 1
    train_data_raw = DataProcessor( store_in['data'],
                                    kw['NbinsPhi'], kw['NbinsRz'],
                                    kw['WindowSize'] )

    train_data = train_data_raw.data_with_boundaries
    #train_data = tf.data.Dataset.from_tensor_slices( train_data_raw )

    for rzslice in train_data:
        #look at the first R/z slice only
        indata = tf.convert_to_tensor(rzslice[:,0], dtype=tf.float32)
        inbins = tf.convert_to_tensor(rzslice[:,1], dtype=tf.float32)

        tcd = TriggerCellDistributor( inshape=indata.shape,
                                      kernel_size=kw['KernelSize'],
                                      window_size=kw['WindowSize'],
                                      phibounds=(kw['MinPhi'],kw['MaxPhi']),
                                      nbinsphi=kw['NbinsPhi'],
                                      rzbounds=(kw['MinROverZ'],kw['MaxROverZ']),
                                      nbinsrz=kw['NbinsRz'] )

        for epoch in range(kw['Epochs']):
            loss, out_dist = tcd.train(indata, inbins)
            print('Epoch {}, Loss: {}'.format(epoch+1, tcd.train_loss.result()))

            
if __name__ == "__main__":
    from airflow.airflow_dag import optimization_kwargs
    for falgo in optimization_kwargs['FesAlgos']:
        optimization( falgo, **optimization_kwargs )
