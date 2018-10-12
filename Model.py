from keras.layers import Lambda, Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Dropout
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from keras import optimizers
import sys
import types
import pandas as pd
import numpy as np
from preprocessing import to_three_d_array

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

######################################################################
# Check if a method has been overridden
######################################################################

def overridden(method):
    """
    returns:
        True if ``method`` overrides some method with the same
        name in a base class.  This is typically used when defining
        abstract base classes or interfaces, to allow subclasses to define
        either of two related methods:

    >>> class EaterI:
    ...     '''Subclass must define eat() or batch_eat().'''
    ...     def eat(self, food):
    ...         if overridden(self.batch_eat):
    ...             return self.batch_eat([food])[0]
    ...         else:
    ...             raise NotImplementedError()
    ...     def batch_eat(self, foods):
    ...         return [self.eat(food) for food in foods]

    """
    # [xx] breaks on classic classes!
    if isinstance(method, types.MethodType) and get_im_class(method) is not None:
        name = method.__name__
        funcs = [cls.__dict__[name]
                 for cls in _mro(get_im_class(method))
                 if name in cls.__dict__]
        return len(funcs) > 1
    else:
        raise TypeError('Expected an instance method.')

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

def create_lagged_df(data, lags):
    data = pd.DataFrame(data)
    df = pd.concat([data.shift(lag) for lag in range(-lags,0)], axis=1)
    df.columns = ['lag {}'.format(-lag) for lag in range(-lags,0)]
    data_combined = data.join(df)
    data_combined = data_combined.fillna(0).values
    return data_combined

def create_callbacks(callbacks):
    if callbacks == None:
        CallBacks = None
    else:
        CallBacks = []
        for CallBack in callbacks:
            if CallBack == "early stopping":
                from keras.callbacks import EarlyStopping
                # Because stochastic gradient descent is noisy, patience must be set to a relative large number
                early_stopping_monitor = EarlyStopping(patience=10)
                CallBacks.append(early_stopping_monitor)

            if CallBack == "TensorBoard":
                from keras.callbacks import TensorBoard
                log_dir = 'C:/Users/zhouyuxuan/PycharmProjects/Masterarbeit/experiments/logdir'
                CallBacks.append(TensorBoard(log_dir=log_dir))
    return CallBacks

# remark: don't need to split the data set inside the function, unless it would not be flexible enough, especially for the case where the data set for online and offline
# mode are totally differently split
def offline_mode(models, train_set, validation_set, num_epochs, learning_rate, callbacks=None):
    CallBacks = create_callbacks(callbacks)
    opt = optimizers.Adam(lr=learning_rate)
    models[0].compile(optimizer=opt, loss='MSE')
    # train the model on train_set, and cross validate on validation_set
    history = models[0].fit(train_set[:,:-1,:], train_set[:,1:,:],  validation_data=(validation_set[:,:-1,:],validation_set[:,1:,:]),
                            epochs=num_epochs, verbose=1, shuffle=False, callbacks=CallBacks)
    loss = [[],[]]
    predictions = []
    loss[0] = history.history['loss']
    loss[1] = history.history["val_loss"]
    predictions.append(models[0].predict(train_set[:,:-1,:]))
    print(len(predictions[0]))
    print(len(predictions))
    predictions.append(models[0].predict(validation_set[:,:-1,:]))
    return models, loss, predictions

def online_mode(models, application_set, num_epochs, learning_rate, callbacks=None):
    CallBacks = create_callbacks(callbacks)
    # opt = optimizers.Adam(lr=learning_rate)
    models[0].compile(optimizer="Adam", loss='MSE')
    # train and predict alternatively on the application set, using a universal validation set for all the updating steps
    loss = [[], []]
    train_losses = []
    predictions = [[], []] # the first element is prediction on training data, the second on unseen data
    # remark, if we really want to avoid overlapping, we can choose to loop with a step size equal to window size, but achtually we
    # don't need to bother, just choose the original time column, then it's fine
    for i in range(len(application_set[0]) - 1):
        # each data set contains pairs of features and labels in 0 and 1 position
        x = application_set[0][i]  # Dimension from outer to inner
        x = np.expand_dims(x, 0)  # restore the dimension after slicing
        y = application_set[1][i]
        y = np.expand_dims(y, 0)
        # extract next row of the lagged matrix, which is one step further than x at every column position
        x_next = application_set[0][i + 1]
        x_next = np.expand_dims(x_next, 0)
        history = models[0].fit(x, y, epochs=num_epochs, verbose=1, shuffle=False,callbacks=CallBacks)
        train_losses.append(history.history['loss'])
        predictions[1].extend(models[0].predict(x_next))
        predictions[0].extend(models[0].predict(x))
    # rolling loss takes the training loss of last epoch for each example
    rolling_loss = np.array(train_losses)[:, (num_epochs-1)]
    loss[0] = rolling_loss
    # train loss take the average value of the losses over the whole data set for each epoch
    train_loss = np.average(train_losses, axis=0).flatten()
    loss[1] = train_loss
    predictions[0] = np.array(predictions[0])
    predictions[1] = np.array(predictions[1])

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
    def _build_model(lags, filter_size):
        """
        return: Keras model instance
        """
        raise NotImplementedError()

    @staticmethod
    def _format_input(data):
        """
        return: formatted input suitable for specific network
        """
        raise NotImplementedError()

    @classmethod
    def _train_and_predict(cls, params, train, validation_set, test_set=None, application_set=None):
        # submodels are e.g. encoder and decoder part of an autoencoder model, use * to recieve potentially multiple outputs
        models = cls._build_model(params.lags, params.filter_size)
        results = [[],[]]
        # the first element is rolling loss, the second training loss
        losses =[[],[]]
        if params.training_mode == "online":
            for file in range(len(train)):
                print(file)
                data = train[file]
                data_combined = create_lagged_df(data, params.lags)
                train_set = cls._format_input(data_combined)

                model, loss, predictions = online_mode(models, train_set, params.num_epochs, params.learning_rate, params.callbacks)
                # take the first column out, or averaging over the window is also ok, because of the overlapping
                # numpy.squeeze() guarantees the dimension suitable for plot functions
                results[0].append(predictions[0][:, 0].squeeze())
                results[1].append(predictions[1][:, 0].squeeze())
                losses[0].append(loss[0])
                losses[1].append(loss[1])

        # offline mode for wavenet
        if params.training_mode == 'offline':
            validation_set = cls._format_input(validation_set)
        # it doesn't work, because every file length varies, can not be arranged as an array
            for file in range(len(train)):
                print(file)
                data = train[file]

                train_set = cls._format_input(data)

                models, loss, predictions = offline_mode(models, train_set, validation_set,
                                                                params.num_epochs, params.learning_rate, params.callbacks)

                losses[0].append(loss[0])
                losses[1].append(loss[1])
                results[0].append(predictions[0].squeeze())
                results[1].append(predictions[1].squeeze())

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
    def _build_model(input_dim, filter_width):
        """
        return: Keras model instance
        """
        # convolutional layer oparameters
        n_filters = 32
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
        x = Dropout(.2)(x)
        x = Dense(1)(x)
        model = Model(history_seq, x)

        return [model]


    @staticmethod
    def _format_input(data):
        """
        the dimension should be formatted to (1, n, 1), where the second dimension are the number of points as features,
        one file here would be only one example
        """
        print(data.shape)
        if not isinstance(data,np.ndarray):
                data = np.array(data)
        print(data.shape)
        data = np.reshape(data, (1, data.shape[0], 1))
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
    def _build_model(lags, filter_size):
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
    def _format_input(data):
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
    def _build_model(input_dim, filter_size):
        """
        return: Keras model instance
        """
        mdl = Sequential()
        # remark: input_shape: tuple, but input_dim: integer(especially for 1 D layer)
        mdl.add(Dense(12, input_dim=input_dim, activation='relu'))
        mdl.add(Dense(6, activation='relu'))
        mdl.add(Dense(1))
        return [mdl]

    @staticmethod
    def _format_input(data):
        """
        return: formatted input suitable for specific network
        """
        data_set = [data[:, 1:], data[:, 0]]

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
    def _build_model(lags, filter_size):
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
    def _format_input(data):
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