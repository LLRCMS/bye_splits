import h5py
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D

class DataProcessor():
    """Prepare the data to serve as input to the net in R/z slices"""
    def __init__(self, data, nbinsPhi, nbinsRz, window_size):
        # data variables' indexes
        assert data.attrs['columns'] == ['Rz', 'phi', 'Rz_bin', 'phi_bin']
        self.rz_idx = 0
        self.phi_idx = 1
        self.rzbin_idx = 2
        self.phibin_idx = 3

        self.data = data
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
        self.data = drop(self.data, idxs)
        self.data_with_boundaries = drop(self.data_with_boundaries, idxs)

    def _split(self, sort_index):
        """
        Creates a list of R/z slices, each ordered by phi.
        """
        self.data = self.data.astype('float32')

        rz_slices = np.unique(self.data[:,sort_index])
        assert len(rz_slices) == self.nbins[1]
        assert rz_slices == np.arange(len(rz_slices))

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

class Architecture(tf.keras.Model):
    """Neural network model definition."""
    def __init__(self, inshape, kernel_size):
        super(TCDistribution, self).__init__()
        self.conv1 = Conv1D( filters=32,
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
        self.dense = Dense( units=inshape[1],
                            activation='relu',
                            use_bias=True )

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

class TCDistribution():
    """Neural net workings"""
    def __init__(self, inshape, kernel_size, window_size,
                 phibounds, nbinsphi,
                 rzbounds, nbinsrz,
                 pars=(1, 0.5, 1)):
        """
        Manages quantities related with the neural model being used.
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

        assert len(pars)==2
        self.pars = pars

    def calc_loss(self, indata, inbins, outdata):
        """
        Calculates the model's loss function. Receives slices in R/z as input.
        Each array value corresponds to a trigger cell.
        """
        assert inbins.min()==0
        assert inbins.max()==self.nbinsphi-1

        # bin the output of the neural network
        outbins = tf.histogram_fixed_width_bins(outdata,
                                                value_range=self.phibounds,
                                                nbins=self.nbinsphi)
        assert outbins.min()==0
        assert outbins.max()==self.nbinsphi-1

        # convert the bin ids coming before 0 (shifted boundaries) to negative ones
        inbins[:np.argmin(inbins)] -= inbins.max()+1
        outbins[:np.argmin(outbins)] -= outbins.max()+1

        # calculate the variance between adjacent bins
        variance_loss = 0
        for ibin in np.unique(outbins)[:-self.boundary_width]:
            idxs = (outbins >= ibin) & (outbins <= ibin+self.boundary_width)
            variance_loss += tf.math.reduce_variance(indata[idxs])

        # calculate the earth-mover's distance between the net output and the original data
        wasserstein_loss = tf.cumsum(indata) - tf.cumsum(outdata)
        wasserstein_loss = tf.abs( wasserstein_loss )
        wasserstein_loss = tf.reduce_sum( wasserstein_loss )

        # replicated boundaries should be the same
        assert indata[inbins<0] == indata[inbins>inbins.max()-self.boundary_width]
        boundary_sanity_loss = tf.abs( outdata[outbins<0] -
                                       outdata[outbins>outbins.max()-self.boundary_width] )

        loss_pars = [ tf.Variable(x, trainable=False) for x in self.pars ]
        return ( loss_pars[0] * variance_loss +
                 loss_pars[1] * wasserstein_loss +
                 loss_pars[2] * boundary_sanity_loss )

def optimization(algo, **kw):
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    store_out = h5py.File(kw['OptimizationOut'], mode='w')

    assert len(store_in.keys()) == 1
    train_data_raw = DataProcessor(store_out['data'], kw['NbinsPhi'], kw['NbinsRz'],
                                   kw['WindowSize'])


    I actually want to try everything at once!!!!!
    #train_data = tf.data.Dataset.from_tensor_slices( train_data_raw )

    # Create an instance of the model
    tcdist = TCDistribution( phi_bounds=(kw['MinPhi'],kw['MaxPhi']),
                             nbinsphi=kw['NbinsPhi'],
                             rz_bounds=(kw['MinROverZ'],kw['MaxROverZ']),
                             nbinsrz=kw['NbinsRz'] )
    model = tcdist.architecture( inshape=trainDataRaw.shape,
                                 window_size=kw['KernelSize'] )

    loss_object = tcdist.calc_loss( train_data.data[:,0], train_data.data[:,1], model)

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    @tf.function
    def train_step(images, labels):
      with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

      train_loss(loss)
      train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
      # training=False is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = model(images, training=False)
      t_loss = loss_object(labels, predictions)

      test_loss(t_loss)
      test_accuracy(labels, predictions)

    for epoch in range(kw['Epochs']):
      # Reset the metrics at the start of the next epoch
      train_loss.reset_states()
      train_accuracy.reset_states()
      test_loss.reset_states()
      test_accuracy.reset_states()

      for images, labels in train_ds:
        train_step(images, labels)

      for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

      print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
      )

if __name__ == "__main__":
    from airflow.airflow_dag import optimization_kwargs
    for falgo in optimization_kwargs['FesAlgos']:
        optimization( falgo, **optimization_kwargs )
