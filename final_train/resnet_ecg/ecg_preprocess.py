import os
import glob
import itertools

import numpy as np
import scipy.signal as sig
import pywt
from scipy.signal import medfilt

#
def ecg_preprocessing(data, wfun, dcmp_levels, chop_levels, fs, removebaseline, normalize):
    #
    data = butterworth_notch(data, cut_off = [49, 51], order = 2, sampling_freq = fs)

    if removebaseline:
        first_filtered = medfilt(data,71)
        second_filtered = medfilt(first_filtered,215)
        data = data - second_filtered

    dcmp_levels = min(dcmp_levels, pywt.dwt_max_level(data.shape[1], pywt.Wavelet(wfun)))

    coeffs = pywt.wavedec(data, wfun, mode='symmetric', level = dcmp_levels, axis = -1)
    #
    coeffs_m = [np.zeros_like(coeffs[idx]) if idx >= -chop_levels  else coeffs[idx] for idx in range(-dcmp_levels- 1, 0)]
    #
    data_recon = pywt.waverec(coeffs_m, wfun, mode='symmetric', axis = -1)
    #
    if normalize:
        data_recon = (data_recon - np.mean(data_recon)) /np.std(data_recon)

    #data_recon = butterworth_high_pass(data_recon, cut_off = 0.5, order = 6, sampling_freq = fs)#cut_off=2

    #
    #data_recon = butterworth_notch(data_recon, cut_off = [49, 51], order = 2, sampling_freq = fs)
    #
    '''
    for k, v in testingSet.items():
        first_filtered = medfilt(v['lead0'],71)
        second_filtered = medfilt(first_filtered,215)
        v['lead0'] = v['lead0'] - second_filtered

        first_filtered = medfilt(v['lead1'],71)
        second_filtered = medfilt(first_filtered,215)
        v['lead1'] = v['lead1'] - second_filtered

    # 测试数据集的去噪情况

    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(testingSet['105']['lead0'][:3000])
    plt.show()
    '''
    return data_recon
#
def butterworth_high_pass(x, cut_off, order, sampling_freq):
    #
    nyq_freq = sampling_freq / 2
    digital_cutoff = cut_off / nyq_freq
    #
    b, a = sig.butter(order, digital_cutoff, btype='highpass')
    y = sig.lfilter(b, a, x, axis = -1)
    #
    return y
#
def butterworth_notch(x, cut_off, order, sampling_freq):
    #
    cut_off = np.array(cut_off)
    nyq_freq = sampling_freq / 2
    digital_cutoff = cut_off / nyq_freq
    #
    b, a = sig.butter(order, digital_cutoff, btype='bandstop')
    y = sig.lfilter(b, a, x, axis = -1)
    #
    return y
    #
