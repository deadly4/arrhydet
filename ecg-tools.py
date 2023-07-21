import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.fft import fft, fftfreq
from IPython.display import display
import seaborn as sns

import wfdb
import neurokit2 as nk

import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

from keras import models
from keras import layers

from sklearn.model_selection import train_test_split

def detect_peaks(ecg_signal, fs):
   '''
   to determine:
   1. R, T, P, Q, S peaks
   2. calculate QRS intervals
   3. calculate mean RR itervals
   4. calculate HR [bpm] 
   '''
   _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
   #_, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=fs, method="peak")
   _, waves_peak = nk.ecg_delineate(ecg_signal, 
                                 rpeaks, 
                                 sampling_rate=fs, 
                                 method="peak", 
                                 show=False, 
                                 show_type='peaks')
   
   peaks = {}

   print("-R peaks: " + str(rpeaks['ECG_R_Peaks']))
   peaks['R_Peaks'] = rpeaks['ECG_R_Peaks']
   print("-T peaks: " + str(waves_peak['ECG_T_Peaks']))
   peaks['T_Peaks'] = waves_peak['ECG_T_Peaks']
   print("-P peaks: " + str(waves_peak['ECG_P_Peaks']))
   peaks['P_Peaks'] = waves_peak['ECG_P_Peaks']
   print("-Q peaks: " + str(waves_peak['ECG_Q_Peaks']))
   peaks['Q_Peaks'] = waves_peak['ECG_Q_Peaks']
   print("-S peaks: " + str(waves_peak['ECG_S_Peaks']))
   peaks['S_Peaks'] = waves_peak['ECG_S_Peaks']

   return peaks
   
def normalization(x, train_stats):
   return (x - train_stats['mean']) / train_stats['std']


def preparation_arrhythmia_dataset():
   ecg_readings = ["qrs_duration","p-r_interval","q-t_interval","t_interval",
                   "p_interval","qrs","T","P","QRST",
                   "J", "heart_rate","q_wave",
                   "r_wave","s_wave","R'_wave", "S'_wave", "diagnosis"]
 
   arrhythmia_df = pd.read_csv("ecg-dataset/data_arrhythmia.csv", usecols= ecg_readings, sep=';')
   arrhythmia_df = arrhythmia_df.drop(arrhythmia_df.query('diagnosis == 1').sample(n=1).index)
   arrhythmia_df = arrhythmia_df.drop(arrhythmia_df.query('diagnosis == 3').sample(n=1).index)
   arrhythmia_df = arrhythmia_df.drop(arrhythmia_df.query('diagnosis == 4').sample(n=1).index)
   arrhythmia_df = arrhythmia_df.drop(arrhythmia_df.query('diagnosis == 5').sample(n=1).index)
   arrhythmia_df = arrhythmia_df.drop(arrhythmia_df.query('diagnosis == 6').sample(n=1).index)
   arrhythmia_df = arrhythmia_df.drop(arrhythmia_df.query('diagnosis == 7').sample(n=1).index)
   arrhythmia_df = arrhythmia_df.drop(arrhythmia_df.query('diagnosis == 9').sample(n=1).index)
   arrhythmia_df = arrhythmia_df.drop(arrhythmia_df.query('diagnosis == 15').sample(n=1).index)         

   print(arrhythmia_df.info())

   arrhythmia_df = arrhythmia_df.dropna()
   arrhythmia_df = arrhythmia_df.sample(frac=1)

   train_dataset, tmp_test_dataset = train_test_split(arrhythmia_df, test_size=0.8)
   train_dataset = train_dataset.replace('?', 0)
   tmp_test_dataset = tmp_test_dataset.replace('?', 0)

   test_dataset, valid_dataset = train_test_split(tmp_test_dataset, test_size=0.2)

   train_dataset = train_dataset.astype(float)
   test_dataset  = test_dataset.astype(float)
   valid_dataset = valid_dataset.astype(float)

   train_stats = train_dataset.describe()
   train_stats.pop("diagnosis")
   train_stats = train_stats.transpose()
   min_std = train_stats['std'][train_stats['std'] != 0].min()
   train_stats['std'] += min_std
   print(train_stats)

   train_labels = train_dataset.pop("diagnosis")
   test_labels  = test_dataset.pop("diagnosis")
   valid_labels = valid_dataset.pop("diagnosis")

   train_labels = pd.get_dummies(train_labels, prefix='arrhytm_instance')
   test_labels  = pd.get_dummies(test_labels,  prefix='arrhytm_instance') 
   valid_labels = pd.get_dummies(valid_labels, prefix='arrhytm_instance')

   print(train_labels)
   print(test_labels)
   print(valid_labels)

   normed_train_dataset = normalization(train_dataset, train_stats)
   normed_test_dataset  = normalization(test_dataset, train_stats)
   normed_valid_dataset = normalization(valid_dataset, train_stats)

   print(normed_train_dataset.head(10))
   print("train_dataset: " + str(normed_train_dataset.shape))
   print("test_dataset: "  + str(normed_test_dataset.shape))
   print("valid_dataset: " + str(normed_valid_dataset.shape))
   print("train_labels: "  + str(train_labels.shape))
   print("test_labels: "   + str(test_labels.shape))
   print("valid_labels: "  + str(valid_labels.shape))

   return normed_train_dataset, normed_test_dataset, normed_valid_dataset, train_labels, test_labels, valid_labels

def build_model_with_two_hidden_layers(train_data):
   model = models.Sequential()
   model.add(layers.Dense(64,  activation = 'relu', input_shape = (train_data.shape[1],)))
   model.add(layers.Dense(32,  activation = 'relu'))
   model.add(layers.Dense(16,  activation = 'relu'))
   model.add(layers.Dense(12, activation  = 'softmax'))
   learning_rate = 0.0001
   optimizer = keras.optimizers.Adam(learning_rate)
   model.compile(loss='categorical_crossentropy', 
                 optimizer = optimizer,
                 metrics=['acc'])
   return model

(normed_train_dataset, normed_test_dataset, normed_valid_dataset, train_labels, test_labels, valid_labels) = preparation_arrhythmia_dataset()

model = build_model_with_two_hidden_layers(normed_train_dataset)
model.summary()

epochs = 1000
batch_size = 11

history = model.fit(
   normed_train_dataset,
   train_labels,
   batch_size = batch_size,
   epochs = epochs,
   verbose = 1,
   shuffle = True,
   steps_per_epoch = int(normed_train_dataset.shape[0] / batch_size),
   validation_data = (normed_valid_dataset, valid_labels),
)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()