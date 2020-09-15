import sys
import os
import numpy as np
#import scipy.io as sio
import random
from decimal import Decimal
import argparse
import csv
import os
import shutil
import gc
import time
import random as rn
import numpy as np
import pandas as pd
import warnings
import csv

import scipy.io as sio
from scipy import signal

from sklearn.metrics import f1_score

from resnet_ecg.ecg_preprocess import ecg_preprocessing
from resnet_ecg.densemodel import Net
from resnet_ecg import attentionmodel
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras.backend as K
from keras.layers import Input
from keras.models import Model, load_model
import keras
import pywt

from keras.layers import LSTM, GRU, TimeDistributed, Bidirectional, LeakyReLU
from keras.layers import Dense, Dropout, Activation, Flatten,  Input, Reshape, CuDNNGRU
from keras.layers import Convolution1D, MaxPool1D, GlobalAveragePooling1D,concatenate,AveragePooling1D
from keras import initializers, regularizers, constraints
from keras.layers import Layer
from keras.layers.normalization import BatchNormalization

warnings.filterwarnings("ignore")

''' '''
config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session)

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)


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

# Usage: python rematch_challenge.py test_file_path
def wavelet(ecg, wavefunc, lv, m, n):  #

    coeff = pywt.wavedec(ecg, wavefunc, mode='sym', level=lv)  #
    # sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0

    for i in range(m, n + 1):
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2 * np.log(len(cD)))
            if cD[j] >= Tr:
                coeff[i][j] = np.sign(cD[j]) - Tr
            else:
                coeff[i][j] = 0

    denoised_ecg = pywt.waverec(coeff, wavefunc)
    return denoised_ecg


def wavelet_db6(sig):
    """
    R J, Acharya U R, Min L C. ECG beat classification using PCA, LDA, ICA and discrete
     wavelet transform[J].Biomedical Signal Processing and Control, 2013, 8(5): 437-448.
    param sig: 1-D numpy Array
    return: 1-D numpy Array
    """
    coeffs = pywt.wavedec(sig, 'db6', level=9)
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6')
    return sig_filt

def read_data_seg(data_path, preprocess=True, fs=500, newFs=256, winSecond=10, winNum=10, n_index=0,pre_type="sym"):
    """ Read data """

    # Fixed params
    n_class = 9
    winSize = winSecond * fs
    new_winSize = winSecond * newFs

    # Paths
    path_signals = data_path # os.path.join(data_path, split)

    # Read time-series data
    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = 12

    X = np.zeros((len(channel_files), new_winSize, n_channels)).astype('float32')

    channel_name = ['V6', 'aVF', 'I', 'V4', 'V2', 'aVL', 'V1', 'II', 'aVR', 'V3', 'III', 'V5']

    for i_ch, fil_ch in enumerate(channel_files[:]):  # tqdm

        if i_ch % 1000 == 0:
            pass#print(i_ch)

        ecg = sio.loadmat(os.path.join(path_signals, fil_ch))
        ecg_length = ecg["I"].shape[1]

        if ecg_length > fs * winNum * winSecond:
            print(" too long !!!", ecg_length)
            ecg_length = fs * winNum * winSecond
        if ecg_length < 4500:
            print(" too short !!!", ecg_length)
            break

        slide_steps = int((ecg_length - winSize) / winSecond)

        if ecg_length <= 4500:
            slide_steps = 0

        ecg_channels = np.zeros((new_winSize, n_channels)).astype('float32')

        for i_n, ch_name in enumerate(channel_name):

            ecg_channels[:, i_n] = signal.resample(ecg[ch_name]
                                                   [:, n_index * slide_steps:n_index * slide_steps + winSize].T
                                                   , new_winSize).T
            if preprocess:
                if pre_type == "sym":
                    ecg_channels[:, i_n] = ecg_preprocessing(ecg_channels[:, i_n].reshape(1, new_winSize), 'sym8', 8, 3,
                                                             newFs, removebaseline=False, normalize=False)[0]
                elif pre_type == "db4":
                    ecg_channels[:, i_n] = wavelet(ecg_channels[:, i_n], 'db4', 4, 2, 4)
                elif pre_type == "db6":
                    ecg_channels[:, i_n] = wavelet_db6(ecg_channels[:, i_n])

                # ecg_channels[:, i_n] = (ecg_channels[:, i_n]-np.mean(ecg_channels[:, i_n]))/np.std(ecg_channels[:, i_n])
            else:
                pass
                print(" no preprocess !!! ")

        X[i_ch, :, :] = ecg_channels

    return X


def read_data_pad_zeros(data_path, split="Train", preprocess=True, fs=500, newFs=256, winSecond=10, winNum=10, n_index=0,
              pre_type="sym"):
    """ Read data """

    # Fixed params
    # n_index = 0
    n_class = 9
    winSize = winSecond * fs
    new_winSize = 23296  # winSecond * newFs
    # Paths
    path_signals = data_path # os.path.join(data_path, split)

    # Read time-series data
    channel_files = os.listdir(path_signals)
    channel_files.sort()
    n_channels = 12  # len(channel_files)

    X = np.zeros((len(channel_files), new_winSize, n_channels)).astype('float32')

    channel_name = ['V6', 'aVF', 'I', 'V4', 'V2', 'aVL', 'V1', 'II', 'aVR', 'V3', 'III', 'V5']

    for i_ch, fil_ch in enumerate(channel_files[:]):  # tqdm

        if i_ch % 1000 == 0:
            pass#print(i_ch)

        ecg = sio.loadmat(os.path.join(path_signals, fil_ch))
        ecg_length = ecg["I"].shape[1]

        if ecg_length > 45500:
            ecg_length = 45500

        ecg_channels = np.zeros((new_winSize, n_channels)).astype('float32')

        for i_n, ch_name in enumerate(channel_name):

            ecg_data = signal.resample(ecg[ch_name][:, :ecg_length].T,
                                       int(ecg[ch_name][:, :ecg_length].shape[1] / 500 * newFs)).T
            if preprocess:
                if pre_type == "sym":
                    ecg_channels[-ecg_data.shape[1]:, i_n] = ecg_preprocessing(ecg_data, 'sym8', 8, 3,
                                                                               newFs, removebaseline=False,
                                                                               normalize=False)[0]

                elif pre_type == "db4":
                    ecg_channels[-ecg_data.shape[1]:, i_n] = wavelet(ecg_data[0], 'db4', 4, 2, 4)
                elif pre_type == "db6":
                    ecg_channels[-ecg_data.shape[1]:, i_n] = wavelet_db6(ecg_data[0])

                # ecg_channels[:, i_n] = (ecg_channels[:, i_n]-np.mean(ecg_channels[:, i_n]))/np.std(ecg_channels[:, i_n])
            else:
                pass
                print(" no preprocess !!! ")

        X[i_ch, :, :] = ecg_channels

    return X


def preprocess_y(labels, y, num_class=10):
    bin_label = np.zeros((len(y), num_class)).astype('int8')
    for i in range(len(y)):
        label_nona = labels.loc[y[i]].dropna()
        for j in range(1, label_nona.shape[0]):
            bin_label[i, int(label_nona[j])] = 1
    return bin_label


def arg_parse():
    """
    Parse arguements

    """
    parser = argparse.ArgumentParser(description='Rematch test of ECG Contest')
    parser.add_argument("--test_path", dest='test_path', help=
                        "the file path of Test Data",
                        default="your test_path", type=str)

    #You need to write your test data path with the argparse parameter.
    #For your convenience when testing with local data, you can write your local test set path to default


    return parser.parse_args()

def predict_attention_onenet(test_path):
    pre_type = "db6"  # "sym"
    num_classes = 10
    len_seg = 23296  # 91s

    # Net Structure
    main_input = Input(shape=(len_seg, 12), dtype='float32', name='main_input')
    x = Convolution1D(12, 3, padding='same')(main_input)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 24, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)

    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 3, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Convolution1D(12, 48, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    cnnout = Dropout(0.2)(x)

    x = Bidirectional(CuDNNGRU(12, input_shape=(2250, 12), return_sequences=True, return_state=False))(cnnout)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    x = AttentionWithContext()(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    main_output = Dense(num_classes, activation='sigmoid')(x)
    model = Model(inputs=main_input, outputs=main_output)

    test_files = os.listdir(test_path)
    test_files.sort()

    test_x = read_data_pad_zeros(test_path, split='Val', preprocess=True, n_index=0, pre_type=pre_type)
    print("attention_one_net test_x shape: ", test_x.shape)

    n_fold = 1
    attention_one_blend_test = np.zeros((len(test_files), n_fold, num_classes)).astype('float32')

    model_path = './official_attention_onenet_model/'

    en_amount = 1
    for seed in range(en_amount):
        print("*********Start Attention One Net Only The Last Fold***************")
        for i in range(n_fold):
            print('fold: ', i + 1, ' training')
            model_name = 'attention_extend_weights-best_k{}_r{}_0802.hdf5'.format(seed, i+2)
            print(model_name)

            # Evaluate best trained model
            model.load_weights(model_path + model_name)
            attention_one_blend_test[:, i, :] = model.predict(test_x)
        ''' 
        labels = pd.read_csv("/media/uuser/data/final_codes/final_run_semi/reference.csv")# " /media/jdcloud/REFERENCE.csv"
        raw_IDs = labels["File_name"].values.tolist()
        IDs = {}
        IDs["sym"] = raw_IDs
        IDs["db4"] = [i + "_db4" for i in raw_IDs]
        IDs["db6"] = [i + "_db6" for i in raw_IDs]
        tr_IDs = np.array(IDs[pre_type])
        ####################################################################################
        X = np.empty((tr_IDs.shape[0], 23296, 12))
        for j, ID in enumerate(tr_IDs):
            X[j, ] = np.load("/media/uuser/data/ysecgtest/training_data_pre/" + ID + ".npy")
        X_tr = X
        del X

        blend_train = model.predict(X_tr)
        del X_tr
        gc.collect()

        csv_path = "./quarter_final/"#ensemble_csv
        pd.DataFrame(blend_train).to_csv(csv_path+"attention_one_net_fold2.csv",index=None)
        '''
    del test_x
    gc.collect()
    print(" predict_attention_onenet OK !!!!!!!!!")
    return attention_one_blend_test

def predict_attention_nets(test_path,train_x=None,test_x=None):

    pre_type = "sym" # "sym"

    input_size = (2560, 12)
    net_num = 10
    inputs_list = [Input(shape=input_size) for _ in range(net_num)]
    outputs = attentionmodel.build_network(inputs_list, 0.5, num_classes=10, block_size=4, relu=False)
    model = Model(inputs=inputs_list, outputs=outputs)

    test_files = os.listdir(test_path)
    test_files.sort()

    #print("*********read data for attention nets******")
    #test_x = [read_data_seg(test_path, preprocess=True, n_index=i, pre_type=pre_type) for i in range(net_num)]

    n_fold = 3
    n_classes = 10
    attention_blend_test = np.zeros((len(test_files), n_fold, n_classes)).astype('float32')

    model_path = './official_attention_model/'

    ####################################################################################
    #blend_train = np.zeros((6500, n_fold, n_classes)).astype('float32')

    en_amount = 1
    for seed in range(en_amount):
        print("*********Start Attention Nets***************")
        for i in range(n_fold):
            print('fold: ', i + 1, ' training')

            model_name = "attention_extend_weights-best_k{}_r{}_0802_30.hdf5".format(seed, i)
            # Evaluate best trained model
            model.load_weights(model_path + model_name)
            attention_blend_test[:, i, :] = model.predict(test_x)

            #blend_train[:,i,:] = model.predict(train_x)

    del test_x
    gc.collect()
    '''  
    train_pd0 = pd.DataFrame(blend_train[:,0,:])
    train_pd1 = pd.DataFrame(blend_train[:,1,:])
    train_pd2 = pd.DataFrame(blend_train[:,2,:])
    csv_path = "./quarter_final/"#quarter_final
    train_pd0.to_csv(csv_path+"attention_10net_fold0.csv",index=None)
    train_pd1.to_csv(csv_path+"attention_10net_fold1.csv",index=None)
    train_pd2.to_csv(csv_path+"attention_10net_fold2.csv",index=None)
    '''
    print(" predict_attention_nets OK !!!!!!!!!")
    return attention_blend_test

def predict_attention_nets_0810(test_path,pre_type="sym",train_x=None,test_x=None):

    input_size = (2560, 12)
    net_num = 10
    inputs_list = [Input(shape=input_size) for _ in range(net_num)]
    outputs = attentionmodel.build_network(inputs_list, 0.5, num_classes=10, block_size=4, relu=False)
    model = Model(inputs=inputs_list, outputs=outputs)

    test_files = os.listdir(test_path)
    test_files.sort()

    #print("*********read data for attention nets******")
    #test_x = [read_data_seg(test_path, preprocess=True, n_index=i, pre_type=pre_type) for i in range(net_num)]

    n_fold = 1
    n_classes = 10
    attention_blend_test = np.zeros((len(test_files), n_fold, n_classes)).astype('float32')

    model_path = './official_attention_model/'

    ####################################################################################
    #blend_train = np.zeros((6500, n_fold, n_classes)).astype('float32')

    en_amount = 1
    for seed in range(en_amount):
        print("*********Start Attention Nets***************")
        for i in range(n_fold):
            print('fold: ', i + 1, ' training')
            model_name = "attention_extend_weights-best_k{}_r{}_0809_30.hdf5".format(seed, i+2)
            print(model_name)

            # Evaluate best trained model
            model.load_weights(model_path + model_name)
            attention_blend_test[:, i, :] = model.predict(test_x)
        ''' 
        blend_train = model.predict(train_x)
        gc.collect()
        csv_path = "./quarter_final/"#quarter_final
        pd.DataFrame(blend_train).to_csv(csv_path+"attention_10net_{}_addori_fold2.csv".format(pre_type),index=None)
        '''
        
    del test_x
    gc.collect()
    print(" predict_attention_nets_0810 OK !!!!!!!!!")
    return attention_blend_test

def predict_dense_nets_onefold(test_path,train_x=None,test_x=None):

    pre_type = "sym" #"sym"

    # Net Structure
    input_size = (2560, 12)
    net_num = 10
    inputs_list = [Input(shape=input_size) for _ in range(net_num)]
    net = Net()
    outputs = net.nnet(inputs_list, 0.5, num_classes=10, attention=False)
    model = Model(inputs=inputs_list, outputs=outputs)

    test_files = os.listdir(test_path)
    test_files.sort()

    #print("*********read data for dense nets******")
    #test_x = [read_data_seg(test_path, preprocess=True, n_index=i, pre_type=pre_type) for i in range(net_num)]

    n_fold = 1
    n_classes = 10
    dense_blend_test = np.zeros((len(test_files), n_fold, n_classes)).astype('float32')

    model_path = './official_densenet_model/'

    en_amount = 1
    for seed in range(en_amount):
        print("*********Start Dense Nets***************")
        for i in range(n_fold):
            print('fold: ', i + 1, ' training')

            # Evaluate best trained model
            model.load_weights(model_path + 'densenet_extend_weights-best_k{}_r{}_f0819.hdf5'.format(seed, i))

            dense_blend_test[:, i, :] = model.predict(test_x)
        ''' 
        blend_train = model.predict(train_x)
        gc.collect()
        csv_path = "./quarter_final/"###ensemble_csv
        pd.DataFrame(blend_train).to_csv(csv_path+"densenet_f0819_10net_fold.csv",index=None)
        '''
    del test_x
    gc.collect()
    print(" predict_dense_nets_onefold OK !!!!!!!!!")
    return dense_blend_test

def predict_dense_nets_kfold(test_path,train_x=None,test_x=None):

    pre_type = "db6" #"sym"

    # Net Structure
    input_size = (2560, 12)
    net_num = 10
    inputs_list = [Input(shape=input_size) for _ in range(net_num)]
    net = Net()
    outputs = net.nnet(inputs_list, 0.5, num_classes=10, attention=False)
    model = Model(inputs=inputs_list, outputs=outputs)

    test_files = os.listdir(test_path)
    test_files.sort()

    #print("*********read data for dense nets******")
    #test_x = [read_data_seg(test_path, preprocess=True, n_index=i, pre_type=pre_type) for i in range(net_num)]

    n_fold = 1
    n_classes = 10
    dense_blend_test = np.zeros((len(test_files), n_fold, n_classes)).astype('float32')

    model_path = './official_densenet_model/'

    en_amount = 1
    for seed in range(en_amount):
        print("*********Start Dense Nets***************")
        for i in range(n_fold):
            print('fold: ', i + 1, ' training')
            model_name = "densenet_extend_weights-best_k{}_r{}_0806_30.hdf5".format(seed, i+2)
            print(model_name)

            # Evaluate best trained model
            model.load_weights(model_path + model_name)

            dense_blend_test[:, i, :] = model.predict(test_x)
        ''' 
        blend_train = model.predict(train_x)
        gc.collect()
        csv_path = "./quarter_final/"#"./ensemble_csv/"
        pd.DataFrame(blend_train).to_csv(csv_path+"densenet_4block_10net_fold2.csv",index=None)
        '''
    del test_x
    gc.collect()
    print(" predict_dense_nets_kfold OK !!!!!!!!!")
    return dense_blend_test


def main():

    args = arg_parse()
    test_path = args.test_path
    print("test_path : ",test_path)

    train_x = None
    ##*************************************************************************##
    print("*************First predict_attention_onenet Start *********************")
    attention_onenet_test = predict_attention_onenet(test_path)
    print("attention_onenet_test shape : ",attention_onenet_test.shape)

    ''''  
    #labels = pd.read_csv("/media/jdcloud/REFERENCE.csv")
    labels = pd.read_csv("/media/uuser/data/final_codes/final_run_semi/reference.csv")
    raw_IDs = labels["File_name"].values.tolist()
    IDs = {}
    IDs["sym"] = raw_IDs
    IDs["db4"] = [i + "_db4" for i in raw_IDs]
    IDs["db6"] = [i + "_db6" for i in raw_IDs]

    pre_type = "sym"
    X = np.empty((6500, 10, 2560, 12))
    for i, ID in enumerate(IDs[pre_type]):
        X[i,] = np.load("/media/uuser/data/ysecgtest/training_data/" + ID + ".npy")
    train_x = [X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 9]]
    del X
    gc.collect()
    '''
    print("*************Second predict_attention_nets Start *********************")
    net_num = 10
    test_x = [read_data_seg(test_path, preprocess=True, n_index=i, pre_type="sym") for i in range(net_num)]
    print("************* read sym test data has done*********************")

    #train_x = None
    attention_nets_test = predict_attention_nets(test_path,train_x=train_x,test_x=test_x)
    print("attention_nets_test shape : ",attention_nets_test.shape)

    pre_type = "sym"
    attention_nets_sym_0810_test = predict_attention_nets_0810(test_path,"sym",train_x=train_x,test_x=test_x)
    print("attention_nets_sym_0810_test shape : ",attention_nets_sym_0810_test.shape)

    #####
    print("*************Third predict_dense_nets_onefold Start *********************")
    dense_nets_onefold = predict_dense_nets_onefold(test_path,train_x=train_x,test_x=test_x)
    print("dense_nets_onefold shape : ",dense_nets_onefold.shape)

    ''' 
    del train_x
    gc.collect()

    pre_type = "db6"
    X = np.empty((6500, 10, 2560, 12))
    for i, ID in enumerate(IDs[pre_type]):
        X[i,] = np.load("/media/uuser/data/ysecgtest/training_data/" + ID + ".npy")
    train_x = [X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7], X[:, 8], X[:, 9]]
    del X
    gc.collect()
    '''
    test_x = [read_data_seg(test_path, preprocess=True, n_index=i, pre_type="db6") for i in range(net_num)]
    print("************* read db6 test data has done*********************")

    print("*************Fourth predict_dense_nets_kfold Start *********************")
    dense_nets_kfold = predict_dense_nets_kfold(test_path,train_x=train_x,test_x=test_x)
    print("dense_nets_kfold shape : ",dense_nets_kfold.shape)

    print("*************Last predict_attention_nets_0810 for db6 data Start *********************")
    attention_nets_db6_0810_test = predict_attention_nets_0810(test_path,"db6",train_x=train_x,test_x=test_x)
    print("attention_nets_db6_0810_test shape : ",attention_nets_db6_0810_test.shape)

    #del train_x
    #gc.collect()

    test = []
    test.append(dense_nets_kfold[:, 0, :])
    test.append(attention_onenet_test[:, 0, :])
    test.append(attention_nets_test[:, 0, :])
    test.append(attention_nets_test[:, 1, :])
    test.append(attention_nets_test[:, 2, :])
    test.append(dense_nets_onefold[:, 0, :])  
    test.append(attention_nets_sym_0810_test[:, 0, :])  
    test.append(attention_nets_db6_0810_test[:, 0, :])  

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    # LR = LogisticRegression(penalty="l2",C=1.0)
    from sklearn.externals import joblib

    out0 = np.hstack(test[:-1])#
    out1 = np.hstack(test[:-2])
    out2 = np.hstack(test[:-2]+[test[-1]])

    LR_clf = joblib.load("LR_ensemble0810.pkl")
    MLkNN8_clf = joblib.load("MLkNN8_ensemble.pkl")
    MLkNN10_clf = joblib.load("MLkNN10_ensemble.pkl")
    MLkNN10_db6_clf = joblib.load("MLkNN10_ensemble_db6_0810.pkl")

    y_pred_LR = LR_clf.predict(out0)
    y_pred_proba_LR = LR_clf.predict_proba(out0)

    y_pred_MLkNN8 = MLkNN8_clf.predict(out1).toarray()
    y_pred_proba_MLkNN8 = MLkNN8_clf.predict_proba(out1).toarray()

    y_pred_MLkNN10 = MLkNN10_clf.predict(out1).toarray()
    y_pred_proba_MLkNN10 = MLkNN10_clf.predict_proba(out1).toarray()

    y_pred_MLkNN10_db6 = MLkNN10_db6_clf.predict(out2).toarray()
    y_pred_proba_MLkNN10_db6 = MLkNN10_db6_clf.predict_proba(out2).toarray()

    y_pred_LR[:,2] = y_pred_MLkNN8[:,2]
    y_pred_proba_LR[:,2] = y_pred_proba_MLkNN8[:,2]

    y_pred_LR[:,5] = y_pred_MLkNN8[:,5]
    y_pred_proba_LR[:,5] = y_pred_proba_MLkNN8[:,5]

    y_pred_LR[:,6] = y_pred_MLkNN8[:,6]
    y_pred_proba_LR[:,6] = y_pred_proba_MLkNN8[:,6]

    y_pred_LR[:,9] = y_pred_MLkNN8[:,9]
    y_pred_proba_LR[:,9] = y_pred_proba_MLkNN8[:,9]

    y_pred_LR[:,0] = y_pred_MLkNN10[:,0]
    y_pred_proba_LR[:,0] = y_pred_proba_MLkNN10[:,0]

    y_pred_LR[:,4] = y_pred_MLkNN10[:,4]
    y_pred_proba_LR[:,4] = y_pred_proba_MLkNN10[:,4]

    y_pred_LR[:,8] = y_pred_MLkNN10[:,8]
    y_pred_proba_LR[:,8] = y_pred_proba_MLkNN10[:,8]

    y_pred_LR[:,7] = y_pred_MLkNN10_db6[:,7]
    y_pred_proba_LR[:,7] = y_pred_proba_MLkNN10_db6[:,7]


    y_pred = y_pred_LR
    y_pred_proba = y_pred_proba_LR

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    pred = []
    for j in range(y_pred.shape[0]):
        pred.append([classes[i] for i in range(10) if y_pred[j][i] == 1])


    for i, val in enumerate(pred):
        if val == []:
            pass
            #for i_p, val_p in enumerate(y_pred_proba[i]):
            #    if val_p >= 0.4:
            #        pred[i].append(i_p)    # f1 == 0.832
                
            if y_pred_proba[i][np.argmax(y_pred_proba[i])] >= 0.4:
                pred[i] = [np.argmax(y_pred_proba[i])]     # f1 == 0.833  0.4

    test_files = os.listdir(test_path)
    test_files.sort()

    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File_name', 'label1', 'label2',
                         'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10'])
        count = 0
        for file_name in test_files:
            if file_name.endswith('.mat'):

                record_name = file_name.strip('.mat')
                answer = []
                answer.append(record_name)

                result = pred[count]

                answer.extend(result)
                for i in range(10 - len(result)):
                    answer.append('')
                count += 1
                writer.writerow(answer)
        csvfile.close()

if __name__ == "__main__":
    main()
