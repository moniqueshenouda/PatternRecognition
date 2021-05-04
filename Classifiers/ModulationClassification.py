from sklearn.preprocessing import LabelBinarizer as LB
from sklearn.preprocessing import normalize
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import pickle

file = open("RML2016.10b.dat", 'rb')
Xd = pickle.load(file, encoding='bytes')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []

for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
file.close()

print(X)
print(lbl)

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
        desired += features[arg]

    return np.stack(desired, axis=1)


# Features Choice
data = extract_features('raw')
labels = np.array(lbl)

in_shape = data[0].shape
out_shape = tuple([1]) + in_shape

np.random.seed(10)
n_examples = labels.shape[0]
r = np.random.choice(range(n_examples), n_examples, replace=False)
train_examples = r[:n_examples * 0.7]
test_examples = r[n_examples * 0.3:]

X_train = data[train_examples]
X_test = data[test_examples]

y_train = LB().fit_transform(labels[train_examples][:, 0])
y_test = LB().fit_transform(labels[test_examples][:, 0])

snr_train = labels[train_examples][:, 1].astype(int)
snr_test = labels[test_examples][:, 1].astype(int)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=in_shape))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Flatten())

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=100, validation_split=0.05, batch_size=2048,
          callbacks=[EarlyStopping(patience=15, restore_best_weights=True)])
