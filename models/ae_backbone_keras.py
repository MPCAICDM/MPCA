from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Flatten, Dense, Conv2DTranspose, Reshape



def Encoder_keras(input_shape=(1, 2, 3), representation_dim=256, representation_activation='tanh',
                 intermediate_activation='relu', hidden_layer_sizes=[32, 64, 128]):

    x_in = Input(shape=input_shape)
    enc = x_in

    # downsample x0.5
    enc = Conv2D(hidden_layer_sizes[0], kernel_size=(5, 5), strides=(2, 2), padding='same')(enc)
    enc = Activation(intermediate_activation)(enc)

    # downsample x0.5
    enc = Conv2D(hidden_layer_sizes[1], kernel_size=(5, 5), strides=(2, 2), padding='same')(enc)
    enc = Activation(intermediate_activation)(enc)

    # downsample x0.5
    enc = Conv2D(hidden_layer_sizes[2], kernel_size=(3, 3), strides=(2, 2),
                 padding='same' if input_shape[1] % 8 == 0 else 'valid')(enc)
    enc = Activation(intermediate_activation)(enc)

    enc = Flatten()(enc)
    rep = Dense(representation_dim, activation=representation_activation)(enc)

    return Model(x_in, rep)

def Decoder_keras(input_shape=(1, 2, 3), hidden_layer_sizes=[32, 64, 128],
                  representation_dim=256, activation='relu'):

    rep_in = Input(shape=(representation_dim,))

    g = Dense(hidden_layer_sizes[2] * int(input_shape[1] / 8) * int(input_shape[1] / 8))(rep_in)
    g = Activation(activation)(g)

    conv_shape = (int(input_shape[1] / 8), int(input_shape[1] / 8), hidden_layer_sizes[2])
    g = Reshape(conv_shape)(g)

    # upsample x2
    g = Conv2DTranspose(hidden_layer_sizes[1], kernel_size=(3, 3), strides=(2, 2),
                        padding='same' if input_shape[1] % 8 == 0 else 'valid')(g)
    g = Activation(activation)(g)

    # upsample x2
    g = Conv2DTranspose(hidden_layer_sizes[0], kernel_size=(5, 5), strides=(2, 2), padding='same')(g)
    g = Activation(activation)(g)


    # upsample x2
    g = Conv2DTranspose(input_shape[2], kernel_size=(5, 5), strides=(2, 2), padding='same')(g)
    g = Activation('tanh')(g)

    return Model(rep_in, g)


def Encoder_Linear_keras(input_shape=(1, 2, 3), representation_dim=256, representation_activation='tanh',
                 intermediate_activation='relu', hidden_layer_sizes=[32, 64, 128]):

    x_in = Input(shape=input_shape)
    enc = x_in

    # downsample x0.5
    enc = Dense(hidden_layer_sizes[0], activation=intermediate_activation)(enc)

    # downsample x0.5
    enc = Dense(hidden_layer_sizes[1], activation=intermediate_activation)(enc)

    # downsample x0.5
    enc = Dense(hidden_layer_sizes[2], activation=intermediate_activation)(enc)

    rep = Dense(representation_dim, activation=representation_activation)(enc)

    return Model(x_in, rep)

def Decoder_Linear_keras(input_shape=(1, 2, 3), representation_dim=256, representation_activation='relu',
                 intermediate_activation='relu', hidden_layer_sizes=[32, 64, 128]):

    x_in = Input(shape=input_shape)
    enc = x_in

    # downsample x0.5
    enc = Dense(hidden_layer_sizes[2], activation=intermediate_activation)(enc)

    # downsample x0.5
    enc = Dense(hidden_layer_sizes[1], activation=intermediate_activation)(enc)

    # downsample x0.5
    enc = Dense(hidden_layer_sizes[0], activation=intermediate_activation)(enc)

    rep = Dense(representation_dim, activation=representation_activation)(enc)

    return Model(x_in, rep)