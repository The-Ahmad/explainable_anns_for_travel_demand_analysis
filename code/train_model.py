#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:40:49 2019

@author: Ahmad Alwosheel
"""

# Training a neural network
# Synthesised and Read (London) datasets can be used

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

import keras
from keras import models
from keras import layers
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# import your data here (either synthesised or real)
import h5py
f = h5py.File('RUM_synth_data_1.h5','r')
input = f['input'].value
Choice = f['choice'].value


# Getting the inputs and the ouputs
y_all = keras.utils.to_categorical(Choice-1, num_classes=3)
x = [0, input.shape[1],input.shape[1]+3 ] 
X_all = np.reshape(input, ( input.shape[0],  x[1], 1))


kfold = StratifiedKFold(n_splits=5, shuffle=True)

input_shape = ( x[1], 1)


# Start neural network
classifier = models.Sequential()
classifier.add(Flatten(input_shape=input_shape))
classifier.add(Dense(units = 4, use_bias=False, activation = 'sigmoid'))
classifier.add(Dense(units = 3, kernel_initializer = 'uniform',use_bias=False, activation ='softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam',
                   loss = 'categorical_crossentropy',
                   metrics = ['accuracy'])

# Fit the ANN (i.e., cross-validation method)
for train, test in kfold.split(X_all, Choice-1):
    classifier.fit(X_all[train], y_all[train], epochs=150, batch_size=64, verbose=0)


# Measure the performance of the trained ANN
scores_all = classifier.evaluate(X_all, y_all, verbose=0)

