from keras.models import Model
from keras.layers import Activation, Convolution1D,Convolution2D, Dropout, GlobalAveragePooling1D,GlobalAveragePooling2D, Concatenate, Dense, Input, AveragePooling1D,AveragePooling2D,LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.layers import Conv1D, BatchNormalization, Activation, AveragePooling1D, Dense
from keras.layers import Dropout, Concatenate, Flatten, Lambda,concatenate
from keras import initializers, regularizers, constraints
from keras.layers import Reshape, CuDNNLSTM, Bidirectional
from keras.layers import Layer
import keras.backend as K
__version__ = '0.0.1'

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

class Net(object):
    def __init__(self):
        pass

    @staticmethod
    def __slice(x, index):
        return x[:, :, index]

    @staticmethod
    def __DenseNet(
        inp,#input_shape=None,
        dense_blocks=3,
        dense_layers=-1,
        growth_rate=12,
        nb_classes=None,
        dropout_rate=None,
        bottleneck=False,
        compression=1.0,
        weight_decay=1e-4,
        depth=40):
        """
        Creating a DenseNet

        Arguments:
            input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
            dense_blocks : amount of dense blocks that will be created (default: 3)
            dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                           or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                           by the given depth (default: -1)
            growth_rate  : number of filters to add per dense block (default: 12)
            nb_classes   : number of classes
            dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                           In the paper the authors recommend a dropout of 0.2 (default: None)
            bottleneck   : (True / False) if true it will be added in convolution block (default: False)
            compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                           of 0.5 (default: 1.0 - will have no compression effect)
            weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
            depth        : number or layers (default: 40)

        Returns:
            Model        : A Keras model instance
        """

        if nb_classes==None:
            raise Exception('Please define number of classes (e.g. num_classes=10). This is required for final softmax.')

        if compression <=0.0 or compression > 1.0:
            raise Exception('Compression have to be a value between 0.0 and 1.0.')

        if type(dense_layers) is list:
            if len(dense_layers) != dense_blocks:
                raise AssertionError('Number of dense blocks have to be same length to specified layers')
        elif dense_layers == -1:
            dense_layers = int((depth - 4)/3)
            if bottleneck:
                dense_layers = int(dense_layers / 2)
            dense_layers = [dense_layers for _ in range(dense_blocks)]
        else:
            dense_layers = [dense_layers for _ in range(dense_blocks)]

        #img_input = Input(shape=input_shape)
        nb_channels = growth_rate

        #print('Creating DenseNet %s' % __version__)
        #print('#############################################')
        #print('Dense blocks: %s' % dense_blocks)
        #print('Layers per dense block: %s' % dense_layers)
        #print('#############################################')

        # Initial convolution layer
        x = Convolution2D(filters=2 * growth_rate, kernel_size=(3,3), padding='same',strides=(1,1),
                          use_bias=False, kernel_regularizer=l2(weight_decay))(inp)

        #x = Convolution1D(filters=growth_rate, kernel_size=32, padding='same',strides=1,
        #                  use_bias=False, kernel_regularizer=l2(weight_decay))(inp)#(img_input)

        # Building dense blocks
        for block in range(dense_blocks - 1):
            #print("block:::",block)
            # Add dense block
            x, nb_channels = Net.__dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck, weight_decay)

            # Add transition_block
            x = Net.__transition_layer(x, nb_channels, dropout_rate, compression, weight_decay)
            nb_channels = int(nb_channels * compression)

        # Add last dense block without transition but for that with global average pooling

        x, nb_channels = Net.__dense_block(x, dense_layers[-1], nb_channels, growth_rate, dropout_rate, weight_decay)
        
        x = BatchNormalization()(x)
        x = Activation('relu')(x)#LeakyReLU(alpha=0.2)(x)#Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        #x = GlobalAveragePooling1D()(x)
        #x = AveragePooling1D(int(x.shape[1]), int(x.shape[1]))(x)
        #x = Dense(nb_classes, activation='softmax')(x)

        return x#Model(img_input, x, name='densenet')

    @staticmethod
    def __dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
        """
        Creates a dense block and concatenates inputs
        """

        x_list = [x]

        for i in range(nb_layers):
            cb = Net.__convolution_block(x, growth_rate, dropout_rate, bottleneck)
            x_list.append(cb)
            x = Concatenate(axis=-1)(x_list)
            nb_channels += growth_rate
        return x, nb_channels

    @staticmethod
    def __convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):
        """
        Creates a convolution block consisting of BN-ReLU-Conv.
        Optional: bottleneck, dropout
        """

        # Bottleneck
        if bottleneck:
            bottleneckWidth = 4
            x = BatchNormalization()(x)
            x = Activation('relu')(x)#LeakyReLU(alpha=0.2)(x)##Activation('relu')(x)
            x = Convolution2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
            #x = Convolution1D(nb_channels * bottleneckWidth, 32, padding='same',use_bias=False, kernel_regularizer=l2(weight_decay))(x)
            # Dropout
            if dropout_rate:
                x = Dropout(dropout_rate)(x)

        # Standard (BN-ReLU-Conv)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)#LeakyReLU(alpha=0.2)(x)#Activation('relu')(x)
        x = Convolution2D(nb_channels, (3, 3), padding='same', use_bias=False)(x)
        #x = Convolution1D(nb_channels, 32, padding='same', use_bias=False)(x)

        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    @staticmethod
    def __transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):
        """
        Creates a transition layer between dense blocks as transition, which do convolution and pooling.
        Works as downsampling.
        """

        x = BatchNormalization()(x)
        x = Activation('relu')(x)#LeakyReLU(alpha=0.2)(x)#Activation('relu')(x)
        x = Convolution2D(int(nb_channels*compression), (1, 1), padding='same',
                          use_bias=False, kernel_regularizer=l2(weight_decay))(x)

        #x = Convolution1D(int(nb_channels*compression), 32, padding='same',strides=2,
        #                  use_bias=False, kernel_regularizer=l2(weight_decay))(x)

        # Adding dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        #x = AveragePooling1D(2,2)(x)
        #
        return x

    @staticmethod
    def __backbone(inp, C=0.001, initial='he_normal'):

        net = Conv1D(4, 31, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(inp)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 11, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(8, 7, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(5, 5)(net)

        net = Conv1D(16, 5, padding='same', kernel_initializer=initial, kernel_regularizer=regularizers.l2(C))(net)
        net = BatchNormalization()(net)
        net = Activation('relu')(net)
        net = AveragePooling1D(int(net.shape[1]), int(net.shape[1]))(net)

        return net

    @staticmethod
    def nnet(inputs, keep_prob, num_classes):

        branches = []
        for i in range(int(len(inputs))):
            ld = inputs[i]#Lambda(Net.__slice, output_shape=(int(inputs.shape[0]), 12), arguments={'index': i})(inputs)
            ld = Reshape((int(2560), 12,1))(ld)
            bch = Net.__DenseNet(inp=ld,nb_classes=9,depth=10,dropout_rate=0.1,growth_rate=16,compression=0.5)#Net.__DenseNet(inp=1d) #Net.__backbone(ld)
            branches.append(bch)
        features = Concatenate(axis=1)(branches);print(features)#features = concatenate(branches,axis=1)#features = Flatten()(features)
        features = Reshape((600,1))(features)
        features = Conv1D(16, 5, padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))(features)
        #features = Bidirectional(CuDNNLSTM(48, input_shape=(features.shape[0],1),return_sequences=True,return_state=False))(features)#CuDNNGRU  CuDNNLSTM
        features = Activation('relu')(features)
        #features = LeakyReLU(alpha=0.3)(features)
        features = Dropout(0.2)(features)
        #features = AttentionWithContext()(features)


        #features = Reshape((600,))(features)

	#x = Activation('relu')(x)#LeakyReLU(alpha=0.2)(x)#Activation('relu')(x)
        #x = Convolution2D(nb_channels, (3, 3), padding='same', use_bias=False)(x)

        #features = Dropout(keep_prob)(features)#, [1,len(inputs),1]
        #print(features);features = Reshape((120,1))(features);print(features)
        #features = Dropout(keep_prob, [1, int(inputs.shape[-1]), 1])(features)
        #features = Bidirectional(CuDNNLSTM(10, return_sequences=True), merge_mode='concat')(features)
        features = Flatten()(features);#print(features)
        net = Dense(units=num_classes, activation='sigmoid')(features)
        return net#, features

