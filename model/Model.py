from keras.layers import Lambda, Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dropout
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras import backend as K
import sys
from model.utils import *
import numpy as np

# Remark: for prediction-based model, they only use the lagged data as feature and the original data as label,
# for reconstruction-based, they also include the original data as feature as well as label
# and the number of features should be even number(multiple of cpus), if it's odd number, then Keras will raise error message

# Python 2/3 compatibility layer. Based on six.

PY3 = sys.version_info[0] == 3

if PY3:
    def get_im_class(meth):
        return meth.__self__.__class__

else:
    def get_im_class(meth):
        return meth.im_class

# for python3, all classes are new style classes, but for python2, there could exist old syle classes whoes type is 'instance'
def _mro(cls):
    """
    Return the method resolution order for ``cls`` -- i.e., a list
    containing ``cls`` and all its base classes, in the order in which
    they would be checked by ``getattr``.  For new-style classes, this
    is just cls.__mro__.  For classic classes, this can be obtained by
    a depth-first left-to-right traversal of ``__bases__``.
    """
    if isinstance(cls, type):
        return cls.__mro__
    else:
        mro = [cls]
        for base in cls.__bases__: mro.extend(_mro(base))
        return mro

# remark: don't need to split the data set inside the function, unless it would not be flexible enough, especially for the case where the data set for online and offline
# mode are totally differently split
def seq2seq_mode(models, application_set, num_epochs, learning_rate, sliding_step, callbacks=None):

    # define inference step
    def predict_sequence(input_sequence):
        history_sequence = input_sequence.copy()
        print("history sequence shape: ", history_sequence.shape)
        pred_sequence = np.zeros((sliding_step, 1))  # initialize output (pred_steps time steps)
        print(pred_sequence.shape)
        for i in range(sliding_step):
            # record next time step prediction (last time step of model output)
            # remark, if direkt indexing one dimension with integer, then this dimension will be reduced
            last_step_pred = models[0].predict(history_sequence)[:,-1:, :]
            print("last step prediction first 10 channels")
            print(last_step_pred.shape)
            pred_sequence[i, 0] = last_step_pred

            # add the next time step prediction to the history sequence
            history_sequence = np.concatenate([history_sequence,
                                                last_step_pred.reshape(1, 1,1)], axis=1)

        return pred_sequence

    CallBacks = create_callbacks(callbacks)
    # opt = optimizers.Adam(lr=learning_rate)
    models[0].compile(optimizer="Adam", loss='MSE')
    # train and predict alternatively on the application set, using a universal validation set for all the updating steps
    loss = [[], []]
    train_losses = []
    predictions = [[], []] # the first element is prediction on training data, the second on unseen data
    # remark, if we really want to avoid overlapping, we can choose to loop with a step size equal to window size, but achtually we
    # don't need to bother, just choose the original time column, then it's fine
    for i in range(int(len(application_set) / sliding_step) - 1):
        # each data set contains pairs of features and labels in 0 and 1 position
        x = application_set[i]  # Dimension from outer to inner
        x = np.expand_dims(x, 0) # restore the dimension after slicing
        # extract next row of the lagged matrix, which is #stride step further than x at every column position
        x_next = application_set[i + sliding_step]
        x_next = np.expand_dims(x_next, 0)
        # Using teacher forcing during training, which means during inference every previous ground truth is fed to the network
        history = models[0].fit(x[:,:-1,:], x[:,-16:,:], epochs=num_epochs, verbose=1, shuffle=False,callbacks=CallBacks)
        train_losses.append(history.history['loss'])
        # Special case(one step ahead prediction): for MLP, each prediction itself is a (1,) array, after squeeze then can not be converted to list
        if sliding_step > 1:
            pred = predict_sequence(x_next[:,:-sliding_step, :])
            predict = list(pred.squeeze())
            predictions[1].extend(predict)
            predict = list(predict_sequence(x[:,:-sliding_step, :]).squeeze())
            predictions[0].extend(predict)
        else:
            predictions[1].extend(predict_sequence(x_next[:,:-sliding_step, :]))
            predictions[0].extend(predict_sequence(x[:,:-sliding_step, :]))
    # convert the list of n (1,) arrays to a (n,1) array
    if sliding_step > 1:
        predictions[0] = np.expand_dims(np.array(predictions[0]), 1)
        predictions[1] = np.expand_dims(np.array(predictions[1]), 1)
    else:
        predictions[0] = np.array(predictions[0])
        predictions[1] = np.array(predictions[1])
    # rolling loss takes the training loss of last epoch for each example
    rolling_loss = np.array(train_losses)[:, (num_epochs-1)]
    loss[0] = rolling_loss
    # train loss take the average value of the losses over the whole data set for each epoch
    train_loss = np.average(train_losses, axis=0).flatten()
    loss[1] = train_loss
    return models, loss, predictions


def online_mode(models, application_set, num_epochs, learning_rate, sliding_step, callbacks=None):
    CallBacks = create_callbacks(callbacks)
    # opt = optimizers.Adam(lr=learning_rate)
    models[0].compile(optimizer="Adam", loss='MSE')
    # train and predict alternatively on the application set, using a universal validation set for all the updating steps
    loss = [[], []]
    train_losses = []
    predictions = [[], []] # the first element is prediction on training data, the second on unseen data
    # remark, if we really want to avoid overlapping, we can choose to loop with a step size equal to window size, but achtually we
    # don't need to bother, just choose the original time column, then it's fine
    for i in range(int(len(application_set[0]) / sliding_step) - 1):
        # each data set contains pairs of features and labels in 0 and 1 position
        x = application_set[0][i]  # Dimension from outer to inner
        x = np.expand_dims(x, 0)  # restore the dimension after slicing
        y = application_set[1][i]
        y = np.expand_dims(y, 0)
        # extract next row of the lagged matrix, which is one step further than x at every column position
        x_next = application_set[0][i + sliding_step]
        x_next = np.expand_dims(x_next, 0)
        history = models[0].fit(x, y, epochs=num_epochs, verbose=1, shuffle=False,callbacks=CallBacks)
        train_losses.append(history.history['loss'])
        # Special case(one step ahead prediction): for MLP, each prediction itself is a (1,) array, after squeeze then can not be converted to list
        if sliding_step > 1:
            pred = models[0].predict(x_next)
            predict = list(models[0].predict(x_next).squeeze())[:sliding_step]
            predictions[1].extend(predict)
            predict = list(models[0].predict(x).squeeze())[:sliding_step]
            predictions[0].extend(predict)
        else:
            predictions[1].extend(models[0].predict(x_next))
            predictions[0].extend(models[0].predict(x))
    # convert the list of n (1,) arrays to a (n,1) array
    if sliding_step > 1:
        predictions[0] = np.expand_dims(np.array(predictions[0]), 1)
        predictions[1] = np.expand_dims(np.array(predictions[1]), 1)
    else:
        predictions[0] = np.array(predictions[0])
        predictions[1] = np.array(predictions[1])
    # rolling loss takes the training loss of last epoch for each example
    rolling_loss = np.array(train_losses)[:, (num_epochs-1)]
    loss[0] = rolling_loss
    # train loss take the average value of the losses over the whole data set for each epoch
    train_loss = np.average(train_losses, axis=0).flatten()
    loss[1] = train_loss
    return models, loss, predictions

# in python2 this will create a new style class object, but in python3 there's no need to do it
class neural_network_model(object):
    """
    A processing interface:

    Subclasses must define:
      - ``_build_model``, ''_format_input''
      - either ``classify()`` or ``classify_many()`` (or both)

    Subclasses may define:
      - either ``prob_classify()`` or ``prob_classify_many()`` (or both)
    """
    @staticmethod
    def _build_model(lags, filter_size, prediction_steps=None):
        """
        return: Keras model instance
        """
        raise NotImplementedError()

    @staticmethod
    def _format_input(data, prediction_steps=None):
        """
        return: formatted input suitable for specific network
        """
        raise NotImplementedError()

    @classmethod
    def _train_and_predict(cls, params, train, general_settings, test_set=None, application_set=None):
        # submodels are e.g. encoder and decoder part of an autoencoder model, use * to recieve potentially multiple outputs
        models = cls._build_model(params.lags, params.filter_size, general_settings.prediction_steps)
        results = [[],[]]
        # the first element is rolling loss, the second training loss
        losses =[[],[]]

        for file in range(len(train)):
            print(file)
            data = train[file]
            data_combined = create_lagged_df(data, params.lags)
            print("data input shape: ", data_combined.shape)
            train_set = cls._format_input(data_combined, general_settings.prediction_steps)
            if general_settings.model_type != "wavenet":
                models, loss, predictions = online_mode(models, train_set, params.num_epochs, params.learning_rate, general_settings.prediction_steps, params.callbacks)
            else:
                models, loss, predictions = seq2seq_mode(models, train_set, params.num_epochs, params.learning_rate,
                                                         general_settings.prediction_steps, params.callbacks)
                # take the first column out, or averaging over the window is also ok, because of the overlapping
                # numpy.squeeze() guarantees the dimension suitable for plot functions
                results[0].append(predictions[0][:, 0].squeeze())
                results[1].append(predictions[1][:, 0].squeeze())
                losses[0].append(loss[0])
                losses[1].append(loss[1])
        return models, losses, results

    # def classify(self, featureset):
    #     """
    #     :return: the most appropriate label for the given featureset.
    #     :rtype: label
    #     """
    #     if overridden(self.classify_many):
    #         return self.classify_many([featureset])[0]
    #     else:
    #         raise NotImplementedError()

class Wavenet(neural_network_model):

    def __init__(self, model, losses, results):
        self.model = model
        self.losses = losses
        self.results = results

    @staticmethod
    def _build_model(input_dim, filter_width, prediction_steps=None):
        """
        return: Keras model instance
        """
        # convolutional layer oparameters
        n_filters = 128
        dilation_rates = [2 ** i for i in range(8)]

        # define an input history series and pass it through a stack of dilated causal convolutions
        history_seq = Input(shape=(None, 1))
        x = history_seq

        for dilation_rate in dilation_rates:
            x = Conv1D(filters=n_filters,
                       kernel_size=filter_width,
                       padding='causal',
                       dilation_rate=dilation_rate)(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(.8)(x)
        x = Dense(64)(x)
        x = Dense(1)(x)

        # extract the last 16 time steps as the training target
        def slice(x, seq_length):
            return x[:, -seq_length:, :]

        pred_seq_train = Lambda(slice, arguments={'seq_length': 16})(x)

        model = Model(history_seq, pred_seq_train)

        return [model]


    @staticmethod
    def _format_input(data, prediction_steps=None):
        # flip the array, in order to get time step from past to current
        data = np.flip(to_three_d_array(data),1)
        return data

    @classmethod
    def train_and_predict(cls, params, train, validation_set, test_set=None, application_set=None):
        models, losses, results = cls._train_and_predict(params, train, validation_set, test_set=None)
        return models, losses, results


class Convolutioanl_autoencoder(neural_network_model):

    def __init__(self, model, losses, results):
        self.model = model
        self.losses = losses
        self.results = results

    @staticmethod
    def _build_model(lags, filter_size, prediction_steps=None):
        """
        return: Keras model instance
        """
        # Can also use None to define a flexible placeholder for input data
        input_dim = lags + 1
        input_segment = Input(shape=(input_dim, 1))

        # Define encoder part
        x = Conv1D(32, filter_size, activation='relu', padding='same')(input_segment)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(16, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(4, filter_size, activation='relu',  padding='same')(x)
        encoded = MaxPooling1D(2, padding='same')(x)

        # Define decoder part
        x = Conv1D(4, filter_size, activation='relu', padding='same')(encoded)
        x = UpSampling1D(2)(x)
        x = Conv1D(16, filter_size, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(32, filter_size, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        decoded = Conv1D(1, filter_size, activation='linear', padding= 'same')(x)
        autoencoder = Model(input_segment, decoded)
        encoder = Model(input_segment, encoded)
        ## TODO
        # how to define the input shape of encoding, anyway to access the input_segement shape?
        # encoded_input = Input(shape=())
        return (autoencoder, encoder)

    @staticmethod
    def _format_input(data, prediction_steps=None):
        """
        return: formatted input suitable for specific network
        """
        data_combined = to_three_d_array(data)
        # for reconstruction-based approach, label and features are the same
        # the first elment is features X, the second is label Y
        data_set = [data_combined, data_combined]

        return data_set

    @classmethod
    def train_and_predict(cls, params, train, validation_set, test_set=None, application_set=None):
        models, losses, results = cls._train_and_predict(params, train, validation_set, test_set=None)
        return models, losses, results

class Multilayer_Perceptron(neural_network_model):

    def __init__(self, model, losses, results):
        self.model = model
        self.losses = losses
        self.results = results

    @staticmethod
    def _build_model(lags, filter_size, prediction_steps=1):
        """
        return: Keras model instance
        """
        input_dim = lags - prediction_steps + 1
        mdl = Sequential()
        # remark: input_shape: tuple, but input_dim: integer(especially for 1 D layer)
        mdl.add(Dense(128, input_dim=input_dim, activation='relu'))
        mdl.add(Dense(64, activation='relu'))
        mdl.add(Dense(prediction_steps))
        return [mdl]

    @staticmethod
    def _format_input(data, prediction_steps=1):
        """
        return: formatted input suitable for specific network
        """
        data_set = [data[:, prediction_steps:], data[:, 0:prediction_steps]]

        return data_set

    @classmethod
    def train_and_predict(cls, params, train, validation_set, test_set=None, application_set=None):
        models, losses, results = cls._train_and_predict(params, train, validation_set, test_set=None)
        return models, losses, results

class Variational_Autoecnoder(neural_network_model):

    def __init__(self, model, losses, results):
        self.model = model
        self.losses = losses
        self.results = results

    @staticmethod
    def _build_model(lags, filter_size, prediction_steps=None):
        """
        return: Keras model instance
        """
        input_dim = lags+1
        def sampling(args):
            """Reparameterization trick by sampling fr an isotropic unit Gaussian.
            # Arguments:
                args (tensor): mean and log of variance of Q(z|X)
            # Returns:
                z (tensor): sampled latent vector
            """

            z_mean, z_log_var = args
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=K.shape(z_mean))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        def vae_loss(x, x_decoded_mean, z_log_var, z_mean):
            mse_loss = K.sum(mse(x, x_decoded_mean), axis=1)
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=[1, 2])
            return K.mean(mse_loss + kl_loss)

        input_segment = Input(shape=(input_dim, 1))

        # Define encoder part
        x = Conv1D(32, filter_size, activation='relu', padding='same')(input_segment)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(16, filter_size, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(4, filter_size, activation='relu',  padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        z_mean = Dense(8)(x)
        z_log_sigma = Dense(8)(x)
        # Remark: this layer will cause the training to fail on the last batch, if the last batch is shorter
        # so a padding trick must be applied
        encoded = Lambda(sampling)([z_mean, z_log_sigma])

        # Define decoder part
        x = Conv1D(4, filter_size, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(16, filter_size, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        x = Conv1D(32, filter_size, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        decoded = Conv1D(1, filter_size, activation='linear', padding= 'same')(x)
        autoencoder = Model(input_segment, decoded)
        vae_losses = vae_loss(input_segment, decoded, z_log_sigma, z_mean)
        autoencoder.add_loss(vae_losses)
        encoder = Model(input_segment, encoded)
        return  (autoencoder, encoder)

    @staticmethod
    def _format_input(data, prediction_steps=None):
        """
        return: formatted input suitable for specific network
        """
        data_combined = to_three_d_array(data)
        # for reconstruction-based approach, label and features are the same
        # the first elment is features X, the second is label Y
        data_set = [data_combined, data_combined]

        return data_set

    @classmethod
    def train_and_predict(cls, params, train, validation_set, test_set=None, application_set=None):
        models, losses, results = cls._train_and_predict(params, train, validation_set, test_set=None)
        return models, losses, results