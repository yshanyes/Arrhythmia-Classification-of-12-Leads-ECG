from tqdm import tqdm
import numpy as np
import pandas as pd
from utils import extract_basic_features

import wfdb
import os
import wfdb.processing as wp
import matplotlib.pyplot as plt
from scipy import signal
from utils import find_noise_features, extract_basic_features
import shutil
import gc
import time
import random as rn
#from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold,StratifiedKFold
#from xgboost import XGBClassifier

import warnings
import scipy.io as sio

from resnet_ecg.utils import one_hot,get_batches
from resnet_ecg.ecg_preprocess import ecg_preprocessing
from resnet_ecg.densemodel import Net

from keras.utils import to_categorical
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,EarlyStopping,ReduceLROnPlateau
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.layers import Input
from keras.models import Model,load_model

path = '/media/jdcloud/'

warnings.filterwarnings("ignore")

config = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
KTF.set_session(session )

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)

class Config(object):
    def __init__(self):
        self.conv_subsample_lengths = [1, 2, 1, 2, 1, 2, 1, 2]
        self.conv_filter_length = 32
        self.conv_num_filters_start = 12
        self.conv_init = "he_normal"
        self.conv_activation = "relu"
        self.conv_dropout = 0.5
        self.conv_num_skip = 2
        self.conv_increase_channels_at = 2
        self.batch_size = 32#128
        self.input_shape = [2560, 12]#[1280, 1]
        self.num_categories = 2

    @staticmethod
    def lr_schedule(epoch):
        lr = 0.1
        if epoch >= 10 and epoch < 20:
            lr = 0.01
        if epoch >= 20:
            lr = 0.001
        print('Learning rate: ', lr)
        return lr


import keras.backend as K


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)



def read_data_seg(data_path, split="Train", preprocess=False, fs=500, newFs=256, winSecond=10, winNum=10, n_index=0):
    """ Read data """

    # Fixed params
    # n_index = 0
    n_class = 9
    winSize = winSecond * fs
    new_winSize = winSecond * newFs
    # Paths
    path_signals = os.path.join(data_path, split)

    # Read labels and one-hot encode
    # label_path = os.path.join(data_path, "reference.txt")
    # labels = pd.read_csv(label_path, sep='\t',header = None)
    # labels = pd.read_csv("reference.csv")

    # Read time-series data
    channel_files = os.listdir(path_signals)
    # print(channel_files)
    channel_files.sort()
    n_channels = 12  # len(channel_files)
    # posix = len(split) + 5

    # Initiate array
    list_of_channels = []

    X = np.zeros((len(channel_files), new_winSize, n_channels))
    i_ch = 0

    channel_name = ['V6', 'aVF', 'I', 'V4', 'V2', 'aVL', 'V1', 'II', 'aVR', 'V3', 'III', 'V5']
    channel_mid_name = ['II', 'aVR', 'V2', 'V5']
    channel_post_name = ['III', 'aVF', 'V3', 'V6']

    for i_ch, fil_ch in tqdm(enumerate(channel_files[:])):  # tqdm
        # print(fil_ch)
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

        ecg_channels = np.zeros((new_winSize, n_channels))

        for i_n, ch_name in enumerate(channel_name):

            ecg_channels[:, i_n] = signal.resample(ecg[ch_name]
                                                   [:, n_index * slide_steps:n_index * slide_steps + winSize].T
                                                   , new_winSize).T
            if preprocess:
                data = ecg_preprocessing(ecg_channels[:, i_n].reshape(1, new_winSize), 'sym8', 8, 3, newFs)
                ecg_channels[:, i_n] = data[0]
            else:
                pass
                ecg_channels[:, i_n] = ecg_channels[:, i_n]

        X[i_ch, :, :] = ecg_channels

    return X

def read_train_data(path):

    ecg12_seg0 = read_data_seg(path, n_index=0)
    ecg12_seg1 = read_data_seg(path, n_index=1)
    ecg12_seg2 = read_data_seg(path, n_index=2)
    ecg12_seg3 = read_data_seg(path, n_index=3)
    ecg12_seg4 = read_data_seg(path, n_index=4)

    ecg12_seg5 = read_data_seg(path, n_index=5)
    ecg12_seg6 = read_data_seg(path, n_index=6)
    ecg12_seg7 = read_data_seg(path, n_index=7)
    ecg12_seg8 = read_data_seg(path, n_index=8)
    ecg12_seg9 = read_data_seg(path, n_index=9)

    X = [ecg12_seg0, ecg12_seg1, ecg12_seg2, ecg12_seg3,
         ecg12_seg4, ecg12_seg5, ecg12_seg6, ecg12_seg7,
         ecg12_seg8, ecg12_seg9,
           ]

    del ecg12_seg0, ecg12_seg1, ecg12_seg2, ecg12_seg3, ecg12_seg4
    del ecg12_seg5, ecg12_seg6, ecg12_seg7, ecg12_seg8, ecg12_seg9

    gc.collect()

    return X

def read_test_data(path):

    test_x_seg0 = read_data_seg(path, split='Val', n_index=0)
    test_x_seg1 = read_data_seg(path, split='Val', n_index=1)
    test_x_seg2 = read_data_seg(path, split='Val', n_index=2)
    test_x_seg3 = read_data_seg(path, split='Val', n_index=3)
    test_x_seg4 = read_data_seg(path, split='Val', n_index=4)

    test_x_seg5 = read_data_seg(path, split='Val', n_index=5)
    test_x_seg6 = read_data_seg(path, split='Val', n_index=6)
    test_x_seg7 = read_data_seg(path, split='Val', n_index=7)
    test_x_seg8 = read_data_seg(path, split='Val', n_index=8)
    test_x_seg9 = read_data_seg(path, split='Val', n_index=9)

    test_x = [test_x_seg0, test_x_seg1, test_x_seg2, test_x_seg3, test_x_seg4,
              test_x_seg5, test_x_seg6, test_x_seg7, test_x_seg8, test_x_seg9,
             ]

    del test_x_seg0, test_x_seg1, test_x_seg2, test_x_seg3, test_x_seg4
    del test_x_seg5, test_x_seg6, test_x_seg7, test_x_seg8, test_x_seg9

    gc.collect()

    return test_x

def preprocess_y(labels,y,num_class=9):
    bin_label = np.zeros((len(y),num_class))
    for i in range(len(y)):
        label_nona = labels.loc[y[i]].dropna()
        for j in range(1,label_nona.shape[0]):
            bin_label[i,int(label_nona[j])]=1
    return bin_label


def add_compile(model, config):
    optimizer = SGD(lr=config.lr_schedule(0), momentum=0.9)  # Adam()#
    model.compile(loss='binary_crossentropy',  # weighted_loss,#'binary_crossentropy',
                  optimizer='adam',  # optimizer,#'adam',
                  metrics=['accuracy', fmeasure, recall, precision])
    # ['accuracy',fbetaMacro,recallMacro,precisionMacro])
    # ['accuracy',fmeasure,recall,precision])

if __name__ == '__main__':

    train_dataset_path = path + "/Train/"
    val_dataset_path = path + "/Val/"

    train_files = os.listdir(train_dataset_path)
    train_files.sort()
    val_files = os.listdir(val_dataset_path)
    val_files.sort()

    labels = pd.read_csv(path+"reference.csv")

    #print(labels.head())
    '''   '''
    inputs0 = Input(shape=(2560,12))
    inputs1 = Input(shape=(2560,12))
    inputs2 = Input(shape=(2560,12))
    inputs3 = Input(shape=(2560,12))
    inputs4 = Input(shape=(2560,12))
    inputs5 = Input(shape=(2560,12))
    inputs6 = Input(shape=(2560,12))
    inputs7 = Input(shape=(2560,12))
    inputs8 = Input(shape=(2560,12))
    inputs9 = Input(shape=(2560,12))

    inputs_list = [inputs0,inputs1,inputs2,inputs3,inputs4,inputs5,inputs6,inputs7,inputs8,inputs9]

    net = Net()
    outputs = net.nnet(inputs_list,0.5,num_classes=9)
    model = Model(inputs =inputs_list,outputs=outputs )

    #print(model.summary())

    bin_label = np.zeros((6500,9))
    for i in range(labels.shape[0]):
        label_nona = labels.loc[i].dropna()
        for j in range(1,label_nona.shape[0]):
            bin_label[i,int(label_nona[j])]=1

    cv_pred_all = 0
    en_amount = 1

    labels_en = pd.read_csv(path + "kfold_labels_en.csv")
    print(labels_en.shape)
    print(labels_en.head())

    data_info = pd.read_csv(path + "data_info.csv")
    print(data_info.head())

    train_index = np.arange(6500)

    label2_list = data_info[data_info.labels_num == 2].index.tolist()
    label3_list = data_info[data_info.labels_num == 3].index.tolist()
    label4_list = data_info[data_info.labels_num == 4].index.tolist()
    label5_list = data_info[data_info.labels_num == 5].index.tolist()
    label6_list = data_info[data_info.labels_num == 6].index.tolist()

    train_index = np.insert(train_index, label2_list, label2_list)  # [145:155]

    train_index = np.insert(train_index, label3_list, label3_list)
    train_index = np.insert(train_index, label3_list, label3_list)

    train_index = np.insert(train_index, label4_list, label4_list)
    train_index = np.insert(train_index, label4_list, label4_list)
    train_index = np.insert(train_index, label4_list, label4_list)

    train_index = np.insert(train_index, label5_list, label5_list)
    train_index = np.insert(train_index, label5_list, label5_list)
    train_index = np.insert(train_index, label5_list, label5_list)
    train_index = np.insert(train_index, label5_list, label5_list)

    train_index = np.insert(train_index, label6_list, label6_list)
    train_index = np.insert(train_index, label6_list, label6_list)
    train_index = np.insert(train_index, label6_list, label6_list)
    train_index = np.insert(train_index, label6_list, label6_list)
    train_index = np.insert(train_index, label6_list, label6_list)

    print(train_index.dtype)

    train_index = train_index.astype(np.int)

    train_index.sort()

    print("train_index shape :",train_index.shape)
    print(train_index)

    train_x = np.array(read_train_data(path))
    test_x = np.array(read_test_data(path))

    print("train_x shape :", train_x.shape)

    model_path = 'model/'

    for seed in range(en_amount):
        print("************************")
        n_fold = 5
        n_classes = 9

        #train_label = train_labels  # train_data['score']
        #train_data_df = train_df[feature_name].astype('float32')  # feature_name  columns

        kfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
        kf = kfold.split(train_index, labels_en['label1'])

        blend_train = np.zeros((len(train_x), n_classes))
        blend_test = np.zeros((len(test_x), n_fold, n_classes))

        # train_data_use = train_data.drop(['uid','score','blk_list_flag'], axis=1)
        # test_data_use = test_data.drop(['uid','blk_list_flag'], axis=1)

        ##train_data_use = train_data_df

        # cv_pred = np.zeros(test_data.shape[0])
        valid_best_l2_all = 0
        #oof = np.zeros(train_index.shape[0])
        #feature_importance_df = pd.DataFrame()
        count = 0

        for i, (index_train, index_valid) in enumerate(kf):
            print('fold: ', i+1, ' training')
            t = time.time()

            index_tr = train_index[index_train]
            index_vld = train_index[index_valid]

            '''  '''

            X_tr = train_x[index_tr]

            X_vld = train_x[index_vld]

            #print(index_tr)

            y_tr = preprocess_y(labels,index_tr)
            y_vld = preprocess_y(labels,index_vld)

            #print(y_tr.shape)
            #print(y_vld.shape)
            #print(y_tr[:10])
            #print(y_vld[:10])

            checkpointer = ModelCheckpoint(filepath=model_path+'densenet_weights-best_k{}_r{}.hdf5',
                                           monitor='val_fmeasure', verbose=1, save_best_only=True,
                                           mode='max')  # val_fmeasure
            reduce = ReduceLROnPlateau(monitor='val_fmeasure', factor=0.5, patience=2, verbose=1, min_delta=1e-4,
                                       mode='max')

            config = Config()
            add_compile(model, config)
    
            model_name = 'resnet21.h5'
            earlystop = EarlyStopping(
                monitor='val_fmeasure',  # 'val_categorical_accuracy',
                patience=10,
            )
            checkpoint = ModelCheckpoint(filepath=model_name,
                                         monitor='val_categorical_accuracy', mode='max',
                                         save_best_only='True')

            lr_scheduler = LearningRateScheduler(config.lr_schedule)

            callback_lists = [checkpointer, reduce]  # [checkpointer,lr_scheduler]#
            # [checkpointer,earlystop,lr_scheduler]
            # [checkpoint, earlystop,lr_scheduler]

            history = model.fit(x=X_tr, y=y_tr, batch_size=64, epochs=20,  # class_weight=cw,#'auto',
                                verbose=1, validation_data=(X_vld, y_vld), callbacks=callback_lists)

            # Evaluate best trained model
            model.load_weights(model_path+'densenet_weights-best_k{}_r{}.hdf5'.format(seed, i))

            test_y = model.predict(test_x)

            K.clear_session()
            gc.collect()
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(config=config)
            K.set_session(sess)

            val_y = model.predict_proba(X_vld)


            blend_train[index_vld, :] = val_y
            blend_test[:, i, :] = test_y

            #f1 = f1_score(y_vld, 1 * val_y > 0.2, average='macro')
            #print("... time passed: {0:.1f}sec , fold F1 : {1:.4f}".format(time.time() - t, f1))

            count += 1

    fold_f1_error = f1_score(bin_labels, 1 * blend_train > 0.4, average='macro')
    print('fold f1 score is {0}'.format(fold_f1_error))

    # F1n,F1a,F1o,F1p,F1 = cinc_f1_score(np.array(oof),np.array(train_labels.values))



































































