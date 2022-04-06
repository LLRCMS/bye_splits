import numpy as np
import h5py
from tqdm import tqdm
import datetime
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv1D
from tensorflow.keras.layers import ReLU, LeakyReLU

from plotter import Plotter

DATAPOINTS = 5000
NBINS = 100
DATAMAX = 2*DATAPOINTS/NBINS
DATAMIN = 0.

# https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class
class Architecture(tf.keras.Model):
    """Neural network model definition."""
    def __init__(self, inshape, kernel_size):
        super().__init__()
        assert len(inshape)==1
        self.inshape = inshape

        self.dense1 = Dense( units=50,
                             activation='relu',
                             name='first dense')

        self.dense2 = Dense( units=self.inshape[0],
                             activation='selu',
                             name='second dense')

    def __call__(self, x):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.reshape(x, shape=(-1, x.shape[0]))
        x = self.dense1(x)
        x = self.dense2(x)
        x = tf.squeeze(x)
        return x

class TriggerCellDistributor(tf.Module):
    """Neural net workings"""
    def __init__( self, indata ):
        """
        Manages quantities related to the neural model being used.
        Args: 
              - kernel_size: Length of convolutional kernels
              - window_size: Number of bins considered for each variance calculation.
              Note this is not the same as number of trigger cells (each bin has
              multiple trigger cells).
        """        
        self.indata = indata

        self.architecture = Architecture(self.indata.shape, self.kernel_size)
        self.model_name = 'data/test_model/'

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def calc_loss(self, originaldata, outdata):
        """
        Calculates the model's loss function. Receives slices in R/z as input.
        Each array value corresponds to a trigger cell.
        Args: -originaldata: similar to originaldata, but after postprocessing (denormalization).
                 To avoid confusions, `originaldata` should not be used in this method.
              -inbins: ordered (ascendent) original phi bins
              -bound_size: number of replicated trigger cells to satisfy boundary conditions
              -outdata: neural net postprocessed phi values output
        """
        asserts = []
        with tf.control_dependencies(asserts):
            equality_loss = tf.reduce_sum( tf.math.square(outdata-originaldata) )
            # loss_pars = [ tf.Variable(x, trainable=False) for x in self.pars ]

        return ( { 'equality_loss': 1 * equality_loss,
                  }
                )

    def train(self, save=False):
        # Reset the metrics at the start of the next epoch
        self.train_loss.reset_states()

        # for n in tf.range(steps): # any future loop within one epoch is done here
        with tf.GradientTape() as tape:
            tape.watch(self.indata)
            # tape.watch(self.inbins)

            prediction = self.architecture(self.indata)
            original_data = self.indata

            losses = self.calc_loss(original_data, prediction)
            loss_sum = tf.reduce_sum(list(losses.values()))

        gradients = tape.gradient(loss_sum, self.architecture.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.architecture.trainable_variables))
        self.train_loss(loss_sum)

        return losses, prediction, gradients, self.architecture.trainable_variables

    def save_architecture_diagram(self, name):
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

    def save_model(self):
        self.architecture.save_weights(self.model_name)

    def load_model(self):
        self.architecture.load_weights(self.model_name)

def save_scalar_logs(writer, scalar_map, epoch):
    """
    Saves tensorflow info for Tensorboard visualization.
    `scalar_map` expects a dict of (scalar_name, scalar_value) pairs.
    """
    with writer.as_default():
        for k,v in scalar_map.items():
            tf.summary.scalar(k, v, step=epoch)

def save_gradient_logs(writer, gradients, train_variables, epoch):
    """
    Saves tensorflow info for Tensorboard visualization.
    `scalar_map` expects a dict of (scalar_name, scalar_value) pairs.
    """
    with writer.as_default():
        # In eager mode, grads does not have name, so we get names from model.trainable_weights
        for weights, grads in zip(train_variables, gradients):
            tf.summary.histogram(
                weights.name.replace(':', '_')+'_grads', data=grads, step=epoch)
            
def optimization(algo, **kw):
    #store_in  = h5py.File(kw['OptimizationIn'],  mode='r')
    plotter = Plotter(**kw)
    
    #assert len(store_in.keys()) == 1
    #train_data = [tf.constant([.6]) for _ in range(2)]#
    npoints_per_bin = tf.random.uniform(shape=np.array([NBINS]),
                                        minval=0,
                                        maxval=int(DATAMAX),
                                        dtype=tf.int32)
    for ib in range(NBINS):
        new_tensor = tf.random.uniform(shape=np.array([npoints_per_bin[ib]]),
                                       minval=ib,
                                       maxval=ib+1.,
                                       dtype=tf.float32)
        if ib==0:
            train_data = new_tensor
        else:
            train_data = tf.concat([train_data, new_tensor], axis=0, name='concat')
    train_data = [train_data]
    plotter.save_orig_data( train_data[0],
                            bins=[x for x in range(0, int(NBINS)+1)],
                            minlength=int(NBINS) )

    train_data[0] -= DATAMIN
    train_data[0] /= DATAMAX-DATAMIN

    #orig_data = postprocess(train_data[0], phi_bounds=(DATAMIN,DATAMAX))

    #_, train_data, boundary_sizes = preprocess(
    #     data=store_in['data'],
    #     nbins_phi=kw['NbinsPhi'],
    #     phi_bounds=(kw['MinPhi'],kw['MaxPhi']),
    #     nbins_rz=kw['NbinsRz'],
    #     window_size=kw['WindowSize']
    # )

    #orig_data = postprocess(train_data[0][:,0], phi_bounds=(kw['MinPhi'],kw['MaxPhi']))
    #plotter.save_orig_data( orig_data )
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'

    for i,rzslice in enumerate(train_data):
        if i>0:        #look at the first R/z slice only
            break
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        tcd = TriggerCellDistributor(
            # indata=tf.convert_to_tensor(rzslice[:,0], dtype=tf.float32),
            indata=rzslice,
            # inbins=tf.convert_to_tensor(rzslice[:,1], dtype=tf.float32),
            # bound_size=boundary_sizes[i],
            kernel_size=kw['KernelSize'],
            # window_size=kw['WindowSize'],
            # phibounds=(kw['MinPhi'],kw['MaxPhi']),
            # nbinsphi=kw['NbinsPhi'],
            # rzbounds=(kw['MinROverZ'],kw['MaxROverZ']),
            # nbinsrz=kw['NbinsRz'],
            # pars=(1., 0., 0., 0.),
            pretrained=kw['Pretrain']
        )
        #tcd.save_architecture_diagram('model{}.png'.format(i))

        for epoch in tqdm(range(kw['Epochs'])):
            should_save = True if epoch%20==0 else False
            dictloss, outdata, gradients, train_vars = tcd.train(save=should_save)
            #dictloss.update({'initial_variance': initial_variance})
            save_scalar_logs(
                writer=summary_writer,
                scalar_map=dictloss,
                epoch=epoch
            )
            save_gradient_logs(
                writer=summary_writer,
                gradients=gradients,
                train_variables=train_vars,
                epoch=epoch
            )
            #print('Epoch {}: {}'.format(epoch+1, tcd.train_loss.result()))
            #print('In: ', train_data)
            #print('Out: ', outdata)
            #print()
            outdata *= DATAMAX-DATAMIN
            outdata += DATAMIN

            plotter.save_gen_data( outdata.numpy(),
                                   bins=[x for x in range(int(DATAMIN), int(DATAMAX))],
                                   minlength=int(DATAMAX) )

        plotter.plot(minval=-1, maxval=DATAMAX+2, density=False, show_html=False)
        #plotter.plot(minval=None, maxval=None, density=False, show_html=False)

if __name__ == "__main__":
    from airflow.airflow_dag import optimization_kwargs
    for falgo in optimization_kwargs['FesAlgos']:
        optimization( falgo, **optimization_kwargs )
