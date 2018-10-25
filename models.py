from tensorflow.keras.layers import Lambda, Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import backend as K

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
    mse_loss = K.sum(mse(x, x_decoded_mean),axis=1)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=[1,2])
    return K.mean(mse_loss + kl_loss)

def vae():
    input_segment = Input(shape=(None, 1))

    # Define encoder part
    x = Conv1D(32, 4, activation='relu', padding='same')(input_segment)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(4, 4, activation='relu',  padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    z_mean = Dense(8)(x)
    z_log_sigma = Dense(8)(x)
    # Remark: this layer will cause the training to fail on the last batch, if the last batch is shorter
    # so a padding trick must be applied
    encoded = Lambda(sampling)([z_mean, z_log_sigma])

    # Define decoder part
    x = Conv1D(4, 4, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 4, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 4, activation='linear', padding= 'same')(x)
    autoencoder = Model(input_segment, decoded)
    vae_losses = vae_loss(input_segment, decoded, z_log_sigma, z_mean)
    autoencoder.add_loss(vae_losses)
    encoder = Model(input_segment, encoded)
    return  autoencoder, encoder

def create_conv_autoencoder():
    # Can also use None to define a flexible placeholder for input data
    input_segment = Input(shape=(64, 1))

    # Define encoder part
    x = Conv1D(32, 4, activation='relu', padding='same')(input_segment)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(4, 4, activation='relu',  padding='same')(x)
    encoded = MaxPooling1D(2, padding='same')(x)

    # Define decoder part
    x = Conv1D(4, 4, activation='relu', padding='same')(encoded)
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 4, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 4, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, 4, activation='linear', padding= 'same')(x)
    autoencoder = Model(input_segment, decoded)
    encoder = Model(input_segment, encoded)
    autoencoder.add_loss(mse)
    ## TODO
    # how to define the input shape of encoding, anyway to access the input_segement shape?
    # encoded_input = Input(shape=())
    return autoencoder, encoder

def create_multi_layer_perceptron(feature_number):
    mdl = Sequential()
    # remark: input_shape: tuple, but input_dim: integer(especially for 1 D layer)
    mdl.add(Dense(12, input_dim=feature_number, activation='relu'))
    mdl.add(Dense(6, activation='relu'))
    mdl.add(Dense(1))
    return mdl