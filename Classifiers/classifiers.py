import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc

# general info :
# snrs values are : [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# mods values are : [b'8PSK', b'AM-DSB', b'BPSK', b'CPFSK', b'GFSK', b'PAM4', b'QAM16', b'QAM64', b'QPSK', b'WBFM']
# Xd is a dictionnary of data, its format is (mod,snr) : ndarray 

Xd = pickle.load(open("/content/drive/My Drive/Colab Notebooks/RML2016.10b.dat",'rb'), encoding='bytes')
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])

# get data & its label
# for simple model : data is sepearated by snr value
# meaning snr of -20 have 10*6000 sample of 2*128 shape
X = []
x_extra = []
lbl = []
lbl_extra = []
for snr in snrs:
  for mod in mods :
    x_extra.append(Xd[(mod,snr)])
    for i in range(Xd[(mod,snr)].shape[0]):
      lbl_extra.append(mod)
  X.append(np.vstack(x_extra[:]))
  x_extra.clear()
  lbl.append(lbl_extra[:])
  lbl_extra.clear()

X= np.array(X)
lbl = np.vstack(lbl)
del(x_extra)
del(lbl_extra)
gc.collect()

# for each snr :
acc_log = []
acc_tree = []
acc_forest = []
for i in range(0,20):
  # splitting data
  training_data, test_data, training_labels, test_labels = train_test_split(X[i],lbl[i],test_size = 0.3,random_state = 42)

  # Data preparation to fit model
  numberOfSample, nx, ny = training_data.shape
  x_train = training_data.reshape((numberOfSample, nx*ny))
  numberOfSample, nx, ny = test_data.shape
  x_test = test_data.reshape((numberOfSample, nx*ny))
  label_encoder = LabelEncoder()
  y_train = label_encoder.fit_transform(training_labels)
  y_test = label_encoder.fit_transform(test_labels)
  
  # Logistic Regression
  print("started logistic for snr of", snrs[i])
  logistic_model = LogisticRegression(multi_class= 'multinomial',solver='newton-cg',max_iter= 2000, n_jobs= 5)
  logistic_model.fit(x_train,y_train)
  y_pred = logistic_model.predict(x_test)
  ac = metrics.accuracy_score(y_test,y_pred)
  acc_log.append(ac)
  # if snrs[i] == 0:
  cm = metrics.confusion_matrix(y_test,y_pred)
  cm_df = pd.DataFrame(cm, index = mods, columns= mods)
  plt.figure(figsize=(5.5,4))
  sns.heatmap(cm_df)
  plt.title(f'lg_cm_snr={snrs[i]}\nAccuracy:{ac}')
  plt.show()
  gc.collect()

  # Random Forest
  print("started forest for snr of", snrs[i])
  forest_model = RandomForestClassifier(min_samples_leaf=20, n_jobs=5, warm_start=True)
  forest_model.fit(x_train,y_train)
  y_pred =  forest_model.predict(x_test)
  ac = metrics.accuracy_score(y_test,y_pred)
  acc_forest.append(ac)
  # if snrs[i]==0:
  cm = metrics.confusion_matrix(y_test,y_pred)
  cm_df = pd.DataFrame(cm, index = mods, columns= mods)
  plt.figure(figsize=(5.5,4))
  sns.heatmap(cm_df)
  plt.title(f'rm_cm_snr={snrs[i]}\nAccuracy:{ac}')
  plt.show()
  gc.collect()

  # Decision Tree
  print("started tree for snr of", snrs[i])
  tree_model = DecisionTreeClassifier(max_depth=200, min_samples_leaf=20)
  tree_model.fit(x_train,y_train)
  y_pred = tree_model.predict(x_test)
  ac = metrics.accuracy_score(y_test, y_pred)
  acc_tree.append(ac)
  # if snrs[i]==0:
  cm = metrics.confusion_matrix(y_test,y_pred)
  cm_df = pd.DataFrame(cm, index = mods, columns= mods)
  plt.figure(figsize=(5.5,4))
  sns.heatmap(cm_df)
  plt.title(f'dt_cm_snr={snrs[i]}\nAccuracy:{ac}')
  plt.show()
  gc.collect()

# calculating average accuracy :
print (f'average of logistic={sum(acc_log)/len(acc_log)}')
print (f'average of forest={sum(acc_forest)/len(acc_forest)}')
print (f'average of tree={sum(acc_tree)/len(acc_tree)}')

# plotting snr vs accuracy of each classifier
plt.figure()
fig1, = plt.plot(snrs,acc_log,'bo')
fig2, = plt.plot(snrs,acc_forest,'r+')
fig3, = plt.plot(snrs,acc_tree,'g^')
plt.legend((fig1, fig2, fig3), ('Logisitc R', 'Random F', 'Decision T'))
plt.show()