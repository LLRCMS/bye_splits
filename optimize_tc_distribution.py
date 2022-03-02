import h5py
import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D

class DataProcessor():
    """Prepare the data to serve as input to the net in R/z slices"""
    def __init__(self, data, nbinsPhi):
        self.data = self._format(data)
        self.data = [ x.astype('float32') for x in self.data ]
        self.data = [ x / nbinsPhi for x in self.data ]
        self.bcond = 3

    @staticmethod
    def _format(data):
        """Create a list of R/z slices, each ordered by phi bin."""
        #print(data.attrs['columns'])
        rz_slices = np.unique(data[:,2])
        assert rz_slices == np.arange(len(rz_slices))
        return [ np.sort(data[:][ data[:,2]==slc ][:,-1]) for slc in rz_slices ]


    def boundary_conditions(self, boundary_depth):
        """
        Pad the original data to ensure boundary conditions over
        its Phi dimension
        """
        self.data = [ np.concatenate((x[boundary_depth:],
                                      x,
                                      x[:boundary_depth]), axis=0)
                      for x in self.data ]

    def __call__(self, boundary_depth):
        self.bd = boundary_depth
        self.boundary_conditions(self.bd)
        return self.data


class TCDistributionModel(tf.keras.Model):
    """Neural netowrk model definition."""
    def __init__(self, inshape, kernel_size):
        super(TCDistributionModel, self).__init__()
        self.conv1 = Conv1D( filters=32,
                             kernel_size=kernel_size,
                             strides=1,
                             padding='same', #'valid' means no padding
                             activation='relu',
                             use_bias=True )
        self.conv2 = Conv1D( filters=32,
                             kernel_size=kernel_size,
                             strides=1,
                             padding='same', #'valid' means no padding
                             activation='relu',
                             use_bias=True )
        self.flatten = Flatten()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        return x

def calc_loss(indata, outdata, kernel_size):
    outroll = [ np.roll(arr, shift=i, axis=0) for i in range(kernel_size) ]
    outroll = np.concatenate( outroll )
    variance_loss = tf.reduce_variance( outroll, axis=1 )
    # np.concatenate([ np.roll(arr, shift=i, axis=0) for i in range(3) ])
    # add some expand_dims
    
    wasserstein_loss = tf.cumsum(indata) - tf.cumsum(outdata)
    wasserstein_loss = tf.abs( wasserstein_loss )
    wasserstein_loss = tf.reduce_sum( wasserstein_loss )

    return lambda1 * variance_loss + lambda2 * wasserstein_loss

def optimization(algo, **kw):
    storeIn  = h5py.File(kw['OptimizationIn'],  mode='r')
    storeOut = h5py.File(kw['OptimizationOut'], mode='w')

    assert len(storeIn.keys()) == 1
    trainDataRaw = DataProcessor(storeIn['data'], kw['NbinsPhi'])
    trainData = tf.data.Dataset.from_tensor_slices( trainDataRaw(kw['KernelSize']-1) )

    # Create an instance of the model
    model = TCDistributionModel( inshape=trainDataRaw.shape,
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
