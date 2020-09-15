from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Convolution1D,Convolution2D, Dropout, GlobalAveragePooling1D,GlobalAveragePooling2D, Concatenate, Dense, Input, AveragePooling1D,AveragePooling2D,LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.layers import Conv1D, BatchNormalization, AveragePooling1D, Dense
from keras.layers import Dropout, Concatenate, Flatten, Lambda,concatenate
from keras import regularizers
from keras.layers import Reshape, CuDNNLSTM, Bidirectional


def _bn_relu(layer, dropout=1, **params):#dropout=0,
    layer = BatchNormalization()(layer)
    layer = Activation(params["conv_activation"])(layer)

    if dropout > 0:
        from keras.layers import Dropout
        layer = Dropout(params["conv_dropout"])(layer)

    return layer

def add_conv_weight(
        layer,
        filter_length,
        num_filters,
        subsample_length=1,
        **params):

    layer = Conv1D(
        filters=num_filters,
        kernel_size=filter_length,
        strides=subsample_length,
        padding='same',
        kernel_initializer=params["conv_init"])(layer)
    return layer


def add_conv_layers(layer, **params):
    for subsample_length in params["conv_subsample_lengths"]:
        layer = add_conv_weight(
                    layer,
                    params["conv_filter_length"],
                    params["conv_num_filters_start"],
                    subsample_length=subsample_length,
                    **params)
        layer = _bn_relu(layer, **params)
    return layer

def resnet_block(
        layer,
        num_filters,
        subsample_length,
        block_index,
        **params):
    from keras.layers import Add 
    from keras.layers import MaxPooling1D
    from keras.layers.core import Lambda

    def zeropad(x):
        y = K.zeros_like(x)
        return K.concatenate([x, y], axis=2)

    def zeropad_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 3
        shape[2] *= 2
        return tuple(shape)

    shortcut = MaxPooling1D(pool_size=subsample_length)(layer)
    zero_pad = (block_index % params["conv_increase_channels_at"]) == 0 \
        and block_index > 0
    if zero_pad is True:
        shortcut = Lambda(zeropad, output_shape=zeropad_output_shape)(shortcut)

    for i in range(params["conv_num_skip"]):
        if not (block_index == 0 and i == 0):
            layer = _bn_relu(
                layer,
                dropout=params["conv_dropout"] if i > 0 else 0,
                **params)
        layer = add_conv_weight(
            layer,
            params["conv_filter_length"],
            num_filters,
            subsample_length if i == 0 else 1,
            **params)
    layer = Add()([shortcut, layer])
    return layer

def get_num_filters_at_index(index, num_start_filters, **params):
    return 2**int(index / params["conv_increase_channels_at"]) \
        * num_start_filters

def add_resnet_layers(layer, **params):
    layer = add_conv_weight(
        layer,
        params["conv_filter_length"],
        params["conv_num_filters_start"],
        subsample_length=1,
        **params)
    layer = _bn_relu(layer, **params)
    for index, subsample_length in enumerate(params["conv_subsample_lengths"]):
        num_filters = get_num_filters_at_index(
            index, params["conv_num_filters_start"], **params)
        layer = resnet_block(
            layer,
            num_filters,
            subsample_length,
            index,
            **params)
    layer = _bn_relu(layer, **params)
    layer = AveragePooling1D(int(layer.shape[1]), int(layer.shape[1]))(layer)
    return layer

def add_output_layer(layer, **params):
    from keras.layers.wrappers import TimeDistributed
    #layer = TimeDistributed(Dense(params["num_categories"]))(layer)
    return Activation('relu')(layer)

def add_compile(model, **params):
    from keras.optimizers import Adam
    optimizer = Adam(
        lr=params["learning_rate"],
        clipnorm=params.get("clipnorm", 1))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

def build_network(inputs,keep_prob,num_classes,**params):

    branches = []
    
    for i in range(int(len(inputs))):
        ld = inputs[i]
        #ld = Reshape((int(2560), 12))(ld)
        bch = add_resnet_layers(ld, **params)
        bch = add_output_layer(bch,**params)
        branches.append(bch)

    features = Concatenate(axis=1)(branches)
    features = Dropout(keep_prob, [1,len(inputs),1])(features)
    features = Flatten()(features)
    net = Dense(units=num_classes, activation='sigmoid')(features)

    return net


