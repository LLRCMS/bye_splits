import h5py
import datetime
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D

from data_processing import preprocess

def are_tensors_equal(t1, t2):
    assert t1.shape == t2.shape
    return tf.math.count_nonzero(tf.math.equal(t1,t2))==t1.shape
        
def tensorflow_assignment(tensor, mask, lambda_op):
    """
    Emulate assignment by creating a new tensor.
    The mask must be 1 where the assignment is intended, 0 everywhere else.
    """
    assert tensor.shape == mask.shape
    other = lambda_op(tensor)
    return tensor * (1 - mask) + other * mask

def tensorflow_wasserstein_1d_loss(indata, outdata):
    """Calculates the 1D earth-mover's distance."""
    loss = tf.cumsum(indata) - tf.cumsum(outdata)
    loss = tf.abs( loss )
    return tf.reduce_sum( loss )

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
        x = tf.cast(x, dtype=tf.float32)
        x = tf.reshape(x, shape=(-1, x.shape[0], 1))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.squeeze(x)
        return x

class TriggerCellDistributor(tf.Module):
    """Neural net workings"""
    def __init__( self, indata, inbins, bound_size,
                  kernel_size, window_size,
                  phibounds, nbinsphi, rzbounds, nbinsrz,
                  pars=(1., 1., 1.) ):
        """
        Manages quantities related to the neural model being used.
        Args: 
              - kernel_size: Length of convolutional kernels
              - window_size: Number of bins considered for each variance calculation.
              Note this is not the same as number of trigger cells (each bin has
              multiple trigger cells).
        """        
        self.indata = indata
        self.boundary_size = bound_size
        self.kernel_size = kernel_size
        self.boundary_width = window_size-1
        self.phibounds, self.nbinsphi = phibounds, nbinsphi
        self.rzbounds, self.nbinsrz = rzbounds, nbinsrz

        self.architecture = Architecture(self.indata.shape, kernel_size)

        assert len(pars)==3
        self.pars = pars

        self.subtract_max = lambda x: x - (tf.math.reduce_max(x)+1)
        # convert bin ids coming before 0 (shifted boundaries) to negative ones
        self.inbins = tensorflow_assignment( tensor=inbins,
                                             mask=tf.concat((tf.ones(self.boundary_size),
                                                             tf.zeros(inbins.shape[0]-self.boundary_size)), axis=0),
                                             lambda_op=self.subtract_max,
                                            )
        
        self.indata_variance = self._calc_local_variance(self.indata, self.inbins)
        self.initial_wasserstein_distance = None

        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def calc_loss(self, outdata):
        """
        Calculates the model's loss function. Receives slices in R/z as input.
        Each array value corresponds to a trigger cell.
        Args: -indata: ordered (ascendent) original phi values
              -inbins: ordered (ascendent) original phi bins
              -bound_size: number of replicated trigger cells to satisfy boundary conditions
              -outdata: neural net phi values output
        """
        opt = {'summarize': 10} #number of entries to print if assertion fails
        asserts = [ tf.debugging.Assert(self.indata.shape == outdata.shape, [self.indata.shape, outdata.shape], **opt) ]
        asserts.append( tf.debugging.Assert(tf.math.reduce_min(self.inbins)==-self.boundary_width, [self.inbins], **opt) )
        asserts.append( tf.debugging.Assert(tf.math.reduce_max(self.inbins)==self.nbinsphi-1, [self.inbins], **opt) )

        # bin the output of the neural network
        outbins = tf.histogram_fixed_width_bins(outdata,
                                                value_range=self.phibounds,
                                                nbins=self.nbinsphi)
        outbins = tf.cast(outbins, dtype=tf.float32)
        asserts.append( tf.debugging.Assert(tf.math.reduce_min(outbins)>=0, [outbins], **opt) )
        asserts.append( tf.debugging.Assert(tf.math.reduce_max(outbins)<=self.nbinsphi-1, [outbins], **opt) )

        # convert bin ids coming before 0 (shifted boundaries) to negative ones
        # if this looks convoluted, blame tensorflow, which still does not support tensor assignment
        # a new tensor must be created instead
        outbins = tensorflow_assignment( tensor=outbins,
                                         mask=tf.concat((tf.ones(self.boundary_size),
                                                         tf.zeros(outbins.shape[0]-self.boundary_size)), axis=0),
                                         lambda_op=self.subtract_max,
                                       )

        # replicated boundaries should be the same
        asserts.append( tf.debugging.Assert(
            condition=are_tensors_equal(self.indata[:self.boundary_size], self.indata[-self.boundary_size:]),
            data=[self.indata[:self.boundary_size],self.indata[-self.boundary_size:]],
            **opt) )
        
        with tf.control_dependencies(asserts):
            variance_loss = self._calc_local_variance(outdata, outbins)

            wasserstein_loss = tensorflow_wasserstein_1d_loss(self.indata, outdata)
            if self.initial_wasserstein_distance is None: #the first time
               self.initial_wasserstein_distance = wasserstein_loss
            wasserstein_loss *= (self.indata_variance/self.initial_wasserstein_distance)

            boundary_sanity_loss = tf.reduce_sum(
                tf.abs( outdata[-self.boundary_size:] - outdata[:self.boundary_size] ) )

            loss_pars = [ tf.Variable(x, trainable=False) for x in self.pars ]

        return ( { 'local_variance_loss': loss_pars[0] * variance_loss,
                   'wasserstein_loss':    loss_pars[1] * wasserstein_loss,
                   'boundary_loss':       loss_pars[2] * boundary_sanity_loss },
                 self.indata_variance,
                 tf.unique(outbins[:-self.boundary_size])[0]
                 )

    def _calc_local_variance(self, data, bins):
        """Calculates the variance between adjacent bins."""
        variance_loss = 0
        unique_bins = tf.unique(bins[:-self.boundary_size])[0]
        for ibin in unique_bins:
            idxs = (bins >= ibin) & (bins <= ibin+self.boundary_width)
            variance_loss += tf.math.reduce_variance(data[idxs])
        return variance_loss

    # @tf.function(
    #     input_signature=(
    #         tf.TensorSpec(shape=[], dtype=tf.float32),
    #         tf.TensorSpec(shape=[], dtype=tf.int32))
    # )
    #@tf.function
    def train(self):
        # Reset the metrics at the start of the next epoch
        self.train_loss.reset_states()

        # for n in tf.range(steps): # any future loop within one epoch is done here
        with tf.GradientTape() as tape:
            tape.watch(self.indata)
            tape.watch(self.inbins)

            prediction = self.architecture(self.indata)
            losses, initial_variance, _ = self.calc_loss(prediction)
            loss_sum = tf.reduce_sum(list(losses.values()))

        gradients = tape.gradient(loss_sum, self.architecture.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        self.train_loss(loss_sum)
        return losses, initial_variance, prediction

    def (self, name):
        """Plots the structure of the used architecture."""
        assert name.split('.')[1] == 'png'
        tf.keras.utils.plot_model(
            self.architecture,
            to_file=name,
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )

def save_scalar_logs(writer, scalar_map, epoch):
    """
    Saves tensorflow info for Tensorboard visualization.
    `scalar_map` expects a dict of (scalar_name, scalar_value) pairs.
    """
    with writer.as_default():
        for k,v in scalar_map.items():
            tf.summary.scalar(k, v, step=epoch)

def optimization(algo, **kw):
    store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    #store_out = h5py.File(kw['OptimizationOut'], mode='w')

    assert len(store_in.keys()) == 1
    _, train_data, boundary_sizes = preprocess(
        data=store_in['data'],
        nbins_phi=kw['NbinsPhi'],
        nbins_rz=kw['NbinsRz'],
        window_size=kw['WindowSize']
    )

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'

    for i,rzslice in enumerate(train_data):
        if i>0:        #look at the first R/z slice only
            break
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        tcd = TriggerCellDistributor(
            indata=tf.convert_to_tensor(rzslice[:,0], dtype=tf.float32),
            inbins=tf.convert_to_tensor(rzslice[:,1], dtype=tf.float32),
            bound_size=boundary_sizes[i],
            kernel_size=kw['KernelSize'],
            window_size=kw['WindowSize'],
            phibounds=(kw['MinPhi'],kw['MaxPhi']),
            nbinsphi=kw['NbinsPhi'],
            rzbounds=(kw['MinROverZ'],kw['MaxROverZ']),
            nbinsrz=kw['NbinsRz'],
            pars=(3., 1., 1.),
        )
        tcd.plot('model{}.png'.format(i))

        for epoch in range(kw['Epochs']):
            dictloss, initial_variance, _ = tcd.train()
            dictloss.update({'initial_variance': initial_variance})
            save_scalar_logs(
                writer=summary_writer,
                scalar_map=dictloss,
                epoch=epoch
            )
            print('Epoch {}, Loss: {}'.format(epoch+1, tcd.train_loss.result()))

if __name__ == "__main__":
    from airflow.airflow_dag import optimization_kwargs
    for falgo in optimization_kwargs['FesAlgos']:
        optimization( falgo, **optimization_kwargs )
