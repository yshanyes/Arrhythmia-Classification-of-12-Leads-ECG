import os

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import signal

from resnet_ecg.ecg_preprocess import ecg_preprocessing

def read_data(data_path, split = "TRAIN",preprocess=False):
    """ Read data """

    # Fixed params
    n_class = 2
    n_steps = 2560

    # Paths
    path_signals = os.path.join(data_path, split)

    # Read labels and one-hot encode
    label_path = os.path.join(data_path, "reference.txt")
    labels = pd.read_csv(label_path, sep='\t',header = None)

    # Read time-series data
    channel_files = os.listdir(path_signals)
    #print(channel_files)
    channel_files.sort()
    n_channels = 12#len(channel_files)
    #posix = len(split) + 5

    # Initiate array
    list_of_channels = []
    X = np.zeros((len(channel_files), n_steps, n_channels))
    i_ch = 0
    for i_ch,fil_ch in enumerate(channel_files):
        #channel_name = fil_ch[:-posix]
        #dat_ = pd.read_csv(os.path.join(path_signals,fil_ch), delim_whitespace = True, header = None)

        ecg = sio.loadmat(os.path.join(path_signals,fil_ch))
        ecg['data'] = signal.resample(ecg['data'].T,2560)
        if preprocess:
            data = ecg_preprocessing(ecg['data'].T, 'sym8', 8, 3, 256)
            X[i_ch,:,:] = data.T#ecg['data']
        else:
            X[i_ch,:,:] = ecg['data']
        
        # Record names
        #list_of_channels.append(channel_name)

        # iterate
        #i_ch += 1

    # Return 
    return X, labels, list_of_channels
def one_hot(labels, n_class = 2):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels].T#y = expansion[:, labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"

    return y
def get_batches(X, y, batch_size = 32):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]
