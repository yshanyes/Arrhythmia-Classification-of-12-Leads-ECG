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
            print(i_ch)

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
            print(i_ch)

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


def preprocess_y(labels, y, num_class=9):
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

def predict_attention_net(test_path):
    pre_type = "db6"  # "sym"
    num_classes = 9
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

    n_fold = 3
    attention_one_blend_test = np.zeros((len(test_files), n_fold, num_classes)).astype('float32')

    model_path = './official_attention_onenet_model/'

    en_amount = 1
    for seed in range(en_amount):
        print("*********Start Attention Net***************")
        for i in range(n_fold):
            print('fold: ', i + 1, ' training')
            # Evaluate best trained model
            model.load_weights(model_path + 'attention_extend_weights-best_k{}_r{}_0608.hdf5'.format(seed, i))

            attention_one_blend_test[:, i, :] = model.predict(test_x)

    del test_x
    gc.collect()

    return attention_one_blend_test


def predict_dense_nets(test_path):
    pre_type = "sym"

    # Net Structure
    input_size = (2560, 12)
    net_num = 10
    inputs_list = [Input(shape=input_size) for _ in range(net_num)]
    net = Net()
    outputs = net.nnet(inputs_list, 0.5, num_classes=9)
    model = Model(inputs=inputs_list, outputs=outputs)

    test_files = os.listdir(test_path)
    test_files.sort()

    print("*********read data for dense nets******")
    test_x = [read_data_seg(test_path, preprocess=True, n_index=i, pre_type=pre_type) for i in range(net_num)]

    n_fold = 3
    n_classes = 9
    dense_blend_test = np.zeros((len(test_files), n_fold, n_classes)).astype('float32')

    model_path = './official_densenet_model/'

    en_amount = 1
    for seed in range(en_amount):
        print("*********Start Dense Nets***************")
        for i in range(n_fold):
            print('fold: ', i + 1, ' training')

            # Evaluate best trained model
            model.load_weights(model_path + 'densenet_extend_weights-best_k{}_r{}.hdf5'.format(seed, i))

            dense_blend_test[:, i, :] = model.predict(test_x)

    del test_x
    gc.collect()

    return dense_blend_test


def main():

    args = arg_parse()
    test_path = args.test_path
    print(test_path)

    attention_blend_test = predict_attention_net(test_path)
    dense_blend_test = predict_dense_nets(test_path)

    # best_threshold = [0.6, 0.5, 0.8, 0.6, 0.5, 0.2, 0.4, 0.4, 0.6]
    # out = 0.1 * attention_blend_test[:, 0, :] \
    #      + 0.1 * attention_blend_test[:, 1, :] \
    #      + 0.8 * attention_blend_test[:, 2, :]

    # best_threshold = [0.5, 0.7, 0.6, 0.2, 0.4, 0.2, 0.5, 0.2, 0.4]
    # out = 0.1 * dense_blend_test[:, 0, :] \
    #      + 0.1 * dense_blend_test[:, 1, :] \
    #      + 0.8 * dense_blend_test[:, 2, :]

    best_threshold = [0.6, 0.6, 0.7, 0.4, 0.5, 0.2, 0.4, 0.2, 0.4]

    thr = np.array([0.7, 0.7, 0.7, 0.7, 0., 0.7, 0.8, 1, 0.7])

    out1 = thr * (0.1 * dense_blend_test[:, 0, :] +
                  0.1 * dense_blend_test[:, 1, :] +
                  0.8 * dense_blend_test[:, 2, :])

    out2 = (1 - thr) * (0.1 * attention_blend_test[:, 0, :] +
                        0.1 * attention_blend_test[:, 1, :] +
                        0.8 * attention_blend_test[:, 2, :])
    out = out1 + out2

    y_pred_test = np.array(
        [[1 if out[i, j] >= best_threshold[j] else 0 for j in range(out.shape[1])] for i in range(len(out))])

    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    test_y = y_pred_test

    y_pred = [[1 if test_y[i, j] >= best_threshold[j] else 0 for j in range(test_y.shape[1])]
              for i in range(len(test_y))]
    pred = []
    for j in range(test_y.shape[0]):
        pred.append([classes[i] for i in range(9) if y_pred[j][i] == 1])

    for i, val in enumerate(pred):
        if 0 in val and len(val) > 1:
            flag = 0
            for j in val:
                if (test_y[i][0] - best_threshold[0]) > (test_y[i][j] - best_threshold[j]):
                    pass
                else:
                    flag = 1
            if flag == 1:
                pred[i] = val[1:]
            else:
                pred[i] = val[0]
        if len(val) == 0:
            pred[i] = [np.argmin(np.abs(best_threshold - out[i]))]

    test_files = os.listdir(test_path)
    test_files.sort()

    with open('answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File_name', 'label1', 'label2',
                         'label3', 'label4', 'label5', 'label6', 'label7', 'label8'])
        count = 0
        for file_name in test_files:
            if file_name.endswith('.mat'):

                record_name = file_name.strip('.mat')
                answer = []
                answer.append(record_name)

                result = pred[count]

                answer.extend(result)
                for i in range(8 - len(result)):
                    answer.append('')
                count += 1
                writer.writerow(answer)
        csvfile.close()


if __name__ == "__main__":
    main()
