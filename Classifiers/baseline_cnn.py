#! pip3 install wget
#import wget
import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn import metrics

# ! --> for running shell command .. run once to download data
#! wget http://opendata.deepsig.io/datasets/2016.10/RML2016.10b.tar.bz2
#! tar jxf RML2016.10b.tar.bz2.1

# Xd is a dictionnary of data:
# its format is (mod,snr) : ndarray of shape(6000,2,128)
# in other word its format is (b'8PSK',2) : ndarray 
Xd = pickle.load(open("/content/drive/My Drive/Colab Notebooks/RML2016.10b.dat",'rb'), encoding='bytes')

# snrs values are : [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# mods values are : [b'8PSK', b'AM-DSB', b'BPSK', b'CPFSK', b'GFSK', b'PAM4', b'QAM16', b'QAM64', b'QPSK', b'WBFM']
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

# get data & its label
# data will be the ndarray of a certain combination of (snr,mod)
# we have 6000*200 sample, each sample have an 2*128 ndarray value
# so basically up to this point X = (1200000, 2, 128)
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
          lbl.append((mod, snr))
X = np.vstack(X)

# features
features = {}
# Raw Time Feature
features['raw'] = X[:, 0], X[:, 1]
# First derivative in time
features['derivative'] = normalize(np.gradient(X[:, 0], axis=1)), normalize(np.gradient(X[:, 1], axis=1))
# Integral in time
features['integral'] = normalize(np.cumsum(X[:, 0], axis=1)), normalize(np.cumsum(X[:, 1], axis=1))
# All Together Feature Space  
def extract_features(*arguments):
    desired = ()
    for arg in arguments:
        desired = desired + features[arg]
    return np.stack(desired, axis=1)

# Init general case
# feature choice -> raw
labels = np.array(lbl)
data = extract_features('raw')

# split data 0.7 train 0.3 test randomly
training_data, test_data, training_labels_snrs, test_labels_snr = 
train_test_split(X, lbl, test_size = 0.3, random_state = 42, stratify = lbl)

# here labels are : (b'AM-DSB', -2)
# we don't want this we want first part only so we take b'AM_DSB' only:
training_labels = [label[0] for label in training_labels_snrs]
test_labels = [label[0] for label in test_labels_snr]

# still we don't get label as string byte so transform into vector of
# 9 zeros and a single 1 indicating type [xxxx1xxx]
label_binarizer = LabelBinarizer()
label_binarizer.fit(training_labels)
y_train = label_binarizer.transform(training_labels)
label_binarizer.fit(test_labels)
y_test = label_binarizer.transform(test_labels)

# not only we want to know the type but also the snr value
snr_train = [label[1] for label in training_labels_snrs]
snr_test = [label[1] for label in test_labels_snr]