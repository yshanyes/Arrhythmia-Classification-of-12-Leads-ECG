import tensorflow as tf

import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU,Concatenate
from keras.layers import Dense, Dropout, Activation, Flatten,  Input, Reshape, GRU, CuDNNGRU,CuDNNLSTM
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D,concatenate,AveragePooling1D
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.models import Model
from keras.utils import multi_gpu_model
from keras import initializers, regularizers, constraints
from keras.layers import Layer
from keras.layers.normalization import BatchNormalization
from keras import regularizers

## example:
# X: input data, whose shape is (72000,12)
# Y: output data, whose shape is  = (9,)
# Y = weighted_predict_for_one_sample_only(X)


def dot_product(x, kernel):

    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def backbone(main_input,block_size,relu):
    # main_input = Input(shape=(2560,12), dtype='float32', name='main_input')
    # x = Activation('relu')(x)
    x = main_input
    for i in range(block_size):
        x = Convolution1D(12, 3, padding='same')(x)

        #x = BatchNormalization()(x)

        if relu:
            x = Activation('relu')(x)
        else:
            x = LeakyReLU(alpha=0.3)(x)
        x = Convolution1D(12, 3, padding='same')(x)

        #x = BatchNormalization()(x)

        if relu:
            x = Activation('relu')(x)
        else:
            x = LeakyReLU(alpha=0.3)(x)

        x = Convolution1D(12, 24, strides = 2, padding='same')(x)

        #x = BatchNormalization()(x)

        if relu:
            x = Activation('relu')(x)
        else:
            x = LeakyReLU(alpha=0.3)(x)
        x = Dropout(0.2)(x)
    '''
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides = 2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    '''
    x = Convolution1D(12, 3, padding='same')(x)

    #x = BatchNormalization()(x)

    if relu:
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)

    #x = BatchNormalization()(x)

    if relu:
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 48, strides=2, padding='same')(x)

    #x = BatchNormalization()(x)

    if relu:
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.3)(x)

    cnnout = Dropout(0.2)(x)

    # print(x.shape)

    x = Bidirectional(CuDNNLSTM(12, input_shape=(x.shape[1],12),return_sequences=True,return_state=False))(cnnout)#CuDNNGRU  CuDNNLSTM
    if relu:
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = AttentionWithContext()(x)

    x = Reshape((1, -1))(x)

    x = BatchNormalization()(x)
    if relu:
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    return x


def build_network(inputs, keep_prob, num_classes, block_size, relu):

    branches = []
    for i in range(int(len(inputs))):
        ld = inputs[i]
        bch = backbone(ld, block_size, relu)
        branches.append(bch)
    # print(branches.shape)
    features = Concatenate(axis=1)(branches)
    # print(features.shape)
    # features = concatenate(branches,axis=1);print(features)#features = Flatten()(features)
    # features = Reshape((600,))(features)
    features = Dropout(keep_prob, [1,len(inputs),1])(features)
    # print(features);features = Reshape((120,1))(features);print(features)
    # features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
    # features = Bidirectional(CuDNNLSTM(10, return_sequences=True), merge_mode='concat')(features)
    features = Flatten()(features)
    # print(features)
    #features = Dense(units=512, activation='relu')(features)
    net = Dense(units=num_classes, activation='sigmoid')(features)
    return net
