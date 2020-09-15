
import numpy as np
import pandas as pd
from scipy import signal
from collections import OrderedDict
from scipy import stats
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from rpeakdetectr.rpeakdetect import detect_beats
from pyEntropy.entropy import sample_entropy, shannon_entropy

from sklearn.model_selection import StratifiedKFold, GroupKFold
from os import listdir
from os.path import isfile, join

# import matplotlib.pylab as plt
# from detector_elgendi import det_elgendi

types = {'N': 0, 'A': 1, 'O': 2, '~': 3}


def plot_2d(stft):
    import matplotlib.pylab as plt
    plt.figure()
    plt.imshow(stft, interpolation=None, aspect='auto')
    plt.show()


def find_first_zero_2d(vec):
    for ind, val in enumerate(vec):
        if val == 0.0:
            return ind
    return vec.shape[0]


def find_first_zero_3d(vec):
    for ind, val in enumerate(vec):
        if val[0] == 0.0:
            return ind
    return vec.shape[0]


def convert_label(labels, orig, replace):

    ind = np.where(labels == orig)[0]
    labels[ind] = replace
    return labels


def one_hot_decode(one_hot):
    labels = np.argmax(one_hot, axis=1)
    return labels


def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot = np.zeros((n_labels,n_unique_labels))
    one_hot[np.arange(n_labels), labels] = 1
    return one_hot


def split_signal(x, length):
    n_seg = np.ceil(x.shape[0]/length).astype(np.int32)

    full = (n_seg-1)*length
    arr = x[:full].reshape(n_seg-1, length)
    arr = np.vstack((arr, x[x.shape[0]-length:]))
    return arr


def pull_labels(x):
    """ Pull labels from a file and

    :author: DS 1.10.2016
    """
    y = x[:, 0].astype(np.int16)
    x = x[:, 1:]
    return x, y


def read_references(path):
    reference = pd.read_csv(path, header=None).values
    records = reference[:, 0]
    labels = reference[:, 1]

    labels = [types[labels[i]] for i in range(labels.shape[0])]

    reference = OrderedDict(zip(records, labels))
    return reference


def lorenz_plot_features(rr, rr_1):
    return np.correlate(rr[1:], rr_1)[0]


def predict_nn_stacked(data, model):
    stft = stft_preprocess(data, 40, 80, 40/2, 9)
    pca = joblib.load('pca.pkl')
    stft = pca.transform(stft)

    predictions = np.array([model.predict(frame.reshape(1, -1)) for frame in stft])
    res = np.mean(predictions, axis=0)
    return res


def malin_coeff(x):
    den = np.diff(x).sum()
    if den == 0:
        den = 0.01
    return np.absolute(x).sum() / den


def frequency_ratio(x, low, high, fs, order):
    p = butter_bandpass_filter(x, low, high, fs, order=order, verbose=0)
    p_energy = np.sum(p ** 2)
    all_energy = np.sum(x ** 2)
    ratio = p_energy/all_energy
    if all_energy == 0:
        return 0
    return ratio


def perdiogram(x, fs, bands=np.array([0.1, 6, 12, 20, 30])):
    N = x.shape[0]
    xdft = np.fft.fft(x)
    xdft = xdft[1:round(N / 2) + 1]
    psdx = (1.0 / (fs * N)) * np.abs(xdft) ** 2
    psdx[2: -1] *= 2
    psdx = 10*np.log10(psdx)

    # bands = np.array([0.1, 5, 8, 12, 20, 30])
    n_seg = np.round(psdx.shape[0] / fs * bands).astype(np.int64) + 1

    band = np.zeros((bands.shape[0] - 1,))

    for n in range(bands.shape[0]-1):
        band[n] = np.sum(psdx[n_seg[n]:n_seg[n + 1]])

    return band



def rr_preprocess(sig):
    predictions = detect_beats(sig, 300)
    if predictions.size > 0:
        rr = np.diff(predictions)
        rr = (rr - 50) / (550)
        rr[rr<0] = 0
        rr[rr>1] = 1
    else:
        rr = np.array([])
    if rr.shape[0] > 60:
        rr = rr[:60]
    return rr


def stft_preprocess(sig, n_fft=40, new_fs=80, hop=20, stack_num=9):

    sig = butter_bandpass_filter(sig, 0.5, 40, 300, order=2)
    sig = signal.resample(sig, int((sig.shape[0] * new_fs)/300))
    sig = sig[20:-20]
    stft = np.abs(signal.spectrogram(sig, nfft=n_fft, nperseg=n_fft, noverlap=int(hop))[2])
    stft = np.array(normalize(stft)).T

    stft_concat = np.zeros((stft.shape[0] - stack_num, stft.shape[1] * stack_num))
    dim = stft.shape[1]
    for s_num in range(stack_num):
        l = - stack_num + s_num
        stft_concat[:, dim*s_num:dim*s_num + dim] = stft[s_num:l, :]
    return stft_concat


def following_correlation(x):
    mod = x.shape[0] % 300
    if mod:
        c = np.corrcoef(x[:-mod].reshape(-1, 300))
    else:
        c = np.corrcoef(x.reshape(-1, 300))
    fc = c.diagonal(1)
    return np.var(fc)


def extract_basic_features(x, num, verbose=0):
    cont = 20
    cont2 = 15
    length = int(x.shape[0] / num)
    signals = [x[(l*num):(l*num)+num] for l in range(length-1)]
    signals.append(x[-num:])
    num_signals = len(signals)

    feat = np.zeros((num_signals, 39))

    for ind in range(num_signals):
        x = signals[ind]
        x = butter_bandpass_filter(x, 1, 25, 300, order=3)
        # predictions, values = det_elgendi(x, best_sig=0, weights=np.array([1]))
        predictions = detect_beats(x, 300)
        if predictions.size > 0:
            while predictions[0] < cont:
                predictions = predictions[1:]
            while predictions[-1] > x.shape[0] - cont:
                predictions = predictions[:-1]
        x = normalize(x.reshape(1, -1))[0]
        # RR
        if predictions.size > 3:
            rr = np.diff(predictions).astype(np.float64)
            rr = normalize(rr.reshape(1, -1))[0]
            rr_1 = np.diff(rr)
            rr_2 = np.diff(rr_1)

        if verbose:
            import matplotlib.pylab as plt
            plt.figure()
            plt.plot(x)
            plt.scatter(predictions, np.ones(predictions.shape[0], ), marker='x',  color='red')
            plt.show(block=False)


        # Signal features
        feat[ind, 0] = np.var(x)
        feat[ind, 1] = stats.skew(x)
        feat[ind, 2] = stats.kurtosis(x)

        if predictions.size > 3:

            feat[ind, 3:7] = perdiogram(x, 300)

            qrs_areas = np.array([np.sum(x[prediction-cont:prediction+cont]) for prediction in predictions])
            qrs_areas2 = np.array([np.sum(x[prediction-cont2:prediction+cont2]) for prediction in predictions])
            qrs_malin = np.array([malin_coeff(x[prediction-cont:prediction+cont]) for prediction in predictions])
            qrs_malin2 = np.array([malin_coeff(x[prediction-cont2:prediction+cont2]) for prediction in predictions])

            qrs_malin = qrs_malin[~np.isnan(qrs_malin)]
            qrs_malin2 = qrs_malin2[~np.isnan(qrs_malin2)]

            n_plus = np.array([np.sum(x[prediction-cont:prediction+cont] > 0) for prediction in predictions])
            n_plus2 = np.array([np.sum(x[prediction-cont2:prediction+cont2] > 0) for prediction in predictions])

            predkosc1 = np.array([np.absolute(np.diff(x[prediction-cont:prediction+cont])[1:]) for prediction in predictions])
            predkosc2 = np.array([np.absolute(np.diff(x[prediction-cont2:prediction+cont2])[1:]) for prediction in predictions])

            pred_max_1 = 0.4 * np.max(predkosc1, axis=1)
            pred_max_2 = 0.4 * np.max(predkosc2, axis=1)

            v40 = np.zeros(pred_max_1.size, )
            v40_2 = np.zeros(pred_max_2.size, )
            for i, pred in enumerate(pred_max_1):
                v40[i] = np.where(predkosc1[i, :] > pred)[0].size/(2*cont)
            for i, pred in enumerate(pred_max_2):
                v40_2[i] = np.where(predkosc2[i, :] > pred)[0].size/(2*cont2)

            feat[ind, 7] = np.mean(qrs_areas)
            feat[ind, 8] = np.max(qrs_areas)

            feat[ind, 9] = np.mean(qrs_areas2)
            feat[ind, 10] = np.max(qrs_areas2)

            feat[ind, 11] = np.mean(qrs_malin)
            feat[ind, 12] = np.max(qrs_malin)

            feat[ind, 13] = np.mean(qrs_malin2)
            feat[ind, 14] = np.max(qrs_malin2)

            feat[ind, 15] = np.mean(n_plus)
            feat[ind, 16] = np.max(n_plus)

            feat[ind, 17] = np.mean(n_plus2)
            feat[ind, 18] = np.max(n_plus2)

            feat[ind, 19] = np.mean(v40)
            feat[ind, 20] = np.max(v40)

            feat[ind, 21] = np.mean(v40_2)
            feat[ind, 22] = np.max(v40_2)

            feat[ind, 23] = frequency_ratio(x, 1, 6, 300, 3)
            feat[ind, 24] = frequency_ratio(x, 1, 10, 300, 3)
            feat[ind, 25] = frequency_ratio(x, 8, 20, 300, 3)
            feat[ind, 26] = frequency_ratio(x, 1, 8, 300, 3)
            feat[ind, 27] = frequency_ratio(x, 12, 25, 300, 3)

            # RR features
            feat[ind, 28] = lorenz_plot_features(rr, rr_1)
            feat[ind, 29] = np.var(rr)
            feat[ind, 30] = np.var(rr_1)
            feat[ind, 31] = np.var(rr_2)

            # Entropy
            rr_1_abs = np.abs(rr_1)+0.0001
            feat[ind, 32] = -rr.dot(np.log(rr))
            feat[ind, 33] = -rr_1_abs.dot(np.log(rr_1_abs))
            se = sample_entropy(rr, 2, tolerance=None)
            feat[ind, 34] = se[0]
            feat[ind, 35] = se[1]
            if np.isnan(feat[ind, 34]) or np.isinf(feat[ind, 34]):
                feat[ind, 34] = 0
            if np.isnan(feat[ind, 35]) or np.isinf(feat[ind, 35]):
                feat[ind, 35] = 0
            feat[ind, 36] = shannon_entropy(rr)
            feat[ind, 37] = shannon_entropy(rr_1)

            # Other
            feat[ind, 38] = following_correlation(x)
            # b = perdiogram(rr, 300)

    return feat


def find_class_examples(references, number):
    ind0 = np.where(references == 0)[0]
    ind1 = np.where(references == 1)[0]
    ind2 = np.where(references == 2)[0]
    ind3 = np.where(references == 3)[0]

    # np.random.shuffle(ind0)
    # np.random.shuffle(ind1)
    # np.random.shuffle(ind2)
    # np.random.shuffle(ind3)

    ind0 = ind0[:number]
    ind1 = ind1[:number]
    ind2 = ind2[:number]
    ind3 = ind3[:number]

    return np.concatenate((ind0, ind1, ind2, ind3))


def butter_bandpass_filter(data, lowcut, highcut, fs, order=3, mode=0, verbose=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')

    if verbose:
        for tap in a:
            print("%.20f," % tap)
        for tap in b:
            print("%.20f," % tap)

        import pylab as plt
        plt.figure()
        plt.clf()
        w, h = signal.freqz(b, a, worN=8000)
        plt.plot((w / np.pi) * fs/2.0, np.absolute(h), linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.title('Frequency Response')
        plt.ylim(-0.05, 1.05)
        plt.show()

    if mode == 0:
        y = signal.filtfilt(b, a, data)
    elif mode == 1:
        y = signal.lfilter(b, a, data)
        y = np.flipud(y)
        y = signal.lfilter(b, a, y)
        y = np.flipud(y)
    elif mode == 2:
        y = signal.lfilter(b, a, data)
    else:
        raise("Wrong mode")
    return y


def rms(x):
    x = x.astype(np.int64)
    return np.sqrt(x.dot(x)/x.size)


def find_noise_features(x):
    # from detector_elgendi import det_elgendi
    x = butter_bandpass_filter(x, 1, 35, 300, order=4, verbose=0)
    # predictions, values = det_elgendi(x, best_sig=0, weights=np.array([1]))
    predictions = detect_beats(x, 300)
    if predictions.size == 0:
        snr = 0
    else:
        snr = np.zeros(predictions.shape[0] - 1, )

    diff = 0
    for i, prediction in enumerate(predictions[1:]):

        rms_sig = rms(x[prediction-12: prediction+12])
        rms_noise = rms(x[prediction - 34:prediction - 12])  # 120 - 60 ms przed QRS
        snr[i] = rms_sig / rms_noise
        snr[i] = 10 * np.log10(snr[i] * snr[i])
        diff += max(x[prediction-12: prediction+12]) - min(x[prediction-12: prediction+12])

    diff = diff / (predictions.shape[0] - 1 )

    fr = frequency_ratio(x, 1, 6, 300, 4)
    fr2 = frequency_ratio(x, 12, 20, 300, 4)

    qrs_found = float(predictions.shape[0])/x.shape[0]
    return np.mean(snr), qrs_found, np.var(x), fr, fr2


def plot_boxplots(features, labels, names=''):
    """ The function visualizes box plots
    for each class, plot for feature
    :author: DS 1.07.2016
    """
    import matplotlib.pylab as plt
    unique_labels = np.unique(labels)

    for feat in range(features.shape[1]):
        box_feat_vector = []
        # Choose data of proper classes
        for unique in unique_labels:
            ind1 = np.where(labels == unique)[0]
            examples = features[ind1, feat]
            box_feat_vector.append(examples)
        plt.figure()
        plt.boxplot(box_feat_vector, notch=True, showmeans=True)
        if names:
            plt.title("Boxplot of feature %s" % names[feat])
        else:
            plt.title("Boxplot of feature %s" % feat)
        plt.xlabel("Class")
        plt.ylabel("Data")
        plt.show()


def matlab_f1_score(ref, ans, verbose=True, details=False):
    assert (ref.shape[0] == ans.shape[0])

    AA = np.zeros((4, 4))

    for n in range(ref.shape[0]):
        rec = ref[n]

        this_answer = ans[n]

        if rec == 0:
            if this_answer == 0:
                AA[0, 0] += 1
            elif this_answer == 1:
                AA[0, 1] += 1
            elif this_answer == 2:
                AA[0, 2] += 1
            elif this_answer == 3:
                AA[0, 3] += 1

        elif rec == 1:
            if this_answer == 0:
                AA[1, 0] += 1
            elif this_answer == 1:
                AA[1, 1] += 1
            elif this_answer == 2:
                AA[1, 2] += 1
            elif this_answer == 3:
                AA[1, 3] += 1

        elif rec == 2:
            if this_answer == 0:
                AA[2, 0] += 1
            elif this_answer == 1:
                AA[2, 1] += 1
            elif this_answer == 2:
                AA[2, 2] += 1
            elif this_answer == 3:
                AA[2, 3] += 1

        elif rec == 3:
            if this_answer == 0:
                AA[3, 0] += 1
            elif this_answer == 1:
                AA[3, 1] += 1
            elif this_answer == 2:
                AA[3, 2] += 1
            elif this_answer == 3:
                AA[3, 3] += 1

    F1n = 2 * AA[0, 0] / (sum(AA[0, :]) + sum(AA[:, 0]))
    F1a = 2 * AA[1, 1] / (sum(AA[1, :]) + sum(AA[:, 1]))
    F1o = 2 * AA[2, 2] / (sum(AA[2, :]) + sum(AA[:, 2]))
    F1p = 2 * AA[3, 3] / (sum(AA[3, :]) + sum(AA[:, 3]))
    F1 = (F1n + F1a + F1o) / 3
    if details:
        print(AA)
    if verbose:
        print('F1 measure for Normal rhythm: ' '%1.4f' % F1n)
        print('F1 measure for AF rhythm: ' '%1.4f' % F1a)
        print('F1 measure for Other rhythm: ' '%1.4f' % F1o)
        print('F1 measure for Noisy recordings: ' '%1.4f' % F1p)
        print('Final F1 measure: ' '%1.4f' % F1)

    return F1


def create_splits(folds):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    path = 'training2017/'
    filenames = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".mat") and f[0] == 'A']
    filenames.sort()
    filenames = np.array(filenames)
    references = read_references('training2017/REFERENCE-v2.csv')
    references = np.array([i[1] for i in list(references.items())])

    test_folds = []
    train_folds = []

    ind = 0
    for train_examples, test_examples in skf.split(filenames, references):
        test_folds.append(test_examples)
        train_folds.append(train_examples)
        ind += 1

    return train_folds, test_folds


def extend_indexes_parts(examples, gr):
    ext = np.array([], dtype=np.int64)
    for train_example in examples:
        ext = np.append(ext, np.where(gr == train_example)[0])
    examples = np.append(examples, ext)
    examples = np.unique(examples)
    return examples


def create_splits_parts(folds, filename):
    skf = StratifiedKFold(n_splits=folds, shuffle=False, random_state=42)
    data = pd.read_csv(filename).values
    groups = data[:, 1]

    path = 'training2017/'
    filenames = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".mat") and f[0] == 'A']
    filenames.sort()
    filenames = np.array(filenames)
    references = read_references('training2017/REFERENCE-v2.csv')
    references = np.array([i[1] for i in list(references.items())])

    test_folds = []
    train_folds = []

    ind = 0
    for train_examples, test_examples in skf.split(filenames, references):
        train_examples = extend_indexes_parts(train_examples, groups)
        test_examples = extend_indexes_parts(test_examples, groups)

        test_folds.append(test_examples)
        train_folds.append(train_examples)
        ind += 1

    return train_folds, test_folds