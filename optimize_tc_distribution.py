import h5py
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D

class DataProcessor():
    """Prepare the data to serve as input to the net in R/z slices"""
    def __init__(self, data, nbinsPhi, nbinsRz):
        self.data, self.data_bins = self._format(data)
        self.data = [ x.astype('float32') for x in self.data ]
        self.data = [ x / nbinsPhi for x in self.data ]
        self.nbinsPhi = nbinsPhi
        self.nbinsRz = nbinsRz
        self.bound_cond_width = 3 

    def _format(self, data):
        """
        Create a list of R/z slices, each ordered by phi.
        The two sorts are done separately for convenience. Note that the relative phi 
        ordering of trigger cells in the arrays is nevertheless kept!
        """
        #['Rz', 'phi', 'Rz_bin', 'phi_bin'] (data.attrs['columns'])
        rz_slices = np.unique(data[:,2])
        assert len(rz_slices) == self.nbinsRz
        assert rz_slices == np.arange(len(rz_slices))

        # The relative ordering is kept, despite the change in index ordering
        # Why? The first trigger cells will all have bin=0, the following bin=1, ...
        phis     = [ np.sort(data[:][ data[:,2]==slc ][:,1]) for slc in rz_slices ]
        phi_bins = [ np.sort(data[:][ data[:,2]==slc ][:,3]) for slc in rz_slices ]
        
        return phis, phi_bins

    def boundary_conditions(self):
        """
        Pad the original data to ensure boundary conditions over its Phi dimension.
        The boundary is done in terms of bins, not single trigger cells.
        `self.bound_cond_width` stands for the number of bins seen to the right and left.
        """
        boundary_right_indexes = [ (x >= self.nbinsPhi-self.bound_cond_width)
                                   for x in self.data_bins ]
        boundary_right = [ x[y] for x,y in zip(self.data,boundary_right_indexes) ]
        
        boundary_left_indexes  = [ (x < self.bound_cond_width)
                                   for x in self.data_bins ]
        boundary_right = [ x[y] for x,y in zip(self.data,boundary_left_indexes) ]
        
        self.data = [ np.concatenate((br,x,bl), axis=0)
                      for br,x,bl in zip(boundary_right,self.data,boundary_left) ]

    def __call__(self, boundary_depth):
        self.bound_cond_width = boundary_depth
        self.boundary_conditions()
        return self.data


class TCDistributionModel():
    def __init__(self, inshape, kernel_size, lambdas=(1, 0.5)):
        self.architecture = Architecture(inshape, kernel_size)

    #GETTER

    def calc_loss(self, indata, outdata, kernel_size):
        outroll = [ np.roll(arr, shift=i, axis=0) for i in range(kernel_size) ]
        outroll = np.concatenate( outroll )
        variance_loss = tf.reduce_variance( outroll, axis=1 )
        # np.concatenate([ np.roll(arr, shift=i, axis=0) for i in range(3) ])
        # add some expand_dims

        wasserstein_loss = tf.cumsum(indata) - tf.cumsum(outdata)
        wasserstein_loss = tf.abs( wasserstein_loss )
        wasserstein_loss = tf.reduce_sum( wasserstein_loss )

        lambda1 = tf.Variable(lambdas[0], trainable=False)
        lambda2 = tf.Variable(lambdas[1], trainable=False)
        return lambda1 * variance_loss + lambda2 * wasserstein_loss

class Architecture(tf.keras.Model):
    """Neural network model definition."""
    def __init__(self, inshape, kernel_size):
        super(TCDistributionModel, self).__init__()
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

def optimization(algo, **kw):
    storeIn  = h5py.File(kw['OptimizationIn'],  mode='r')
    storeOut = h5py.File(kw['OptimizationOut'], mode='w')

    assert len(storeIn.keys()) == 1
    trainDataRaw = DataProcessor(storeIn['data'], kw['NbinsPhi'], kw['NbinsRz'])
    trainData = tf.data.Dataset.from_tensor_slices( trainDataRaw(kw['BoundaryWidth']) )

    # Create an instance of the model
    tcdist = TCDistributionModel( ... )
    model = tcdist.architecture( inshape=trainDataRaw.shape,
                                 kernel_size=kw['KernelSize'] )

    loss_object = calc_loss(...) #tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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
