#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 17:32:39 2019

@author: Ahmad Alwosheel
"""

# For RUM synthesised data
# # TT beta is zero

from keras import activations
from keras import backend as K
from scipy.ndimage.filters import median_filter
import keras
import numpy as np
import pandas as pd

import scipy.misc
import time
import os
import h5py

import innvestigate
import innvestigate.utils as iutils
from sklearn.preprocessing import MinMaxScaler


# load your data
f = h5py.File('RUM_synth_data_1.h5', 'r')
input_1 = f['input'].value
choice = f['choice'].value


x = [0, input_1.shape[1] , input_1.shape[1]+3 ] 

y_all = keras.utils.to_categorical(choice-1, num_classes=3)


X_all = input_1
X_all = np.reshape(X_all, ( X_all.shape[0],  x[1], 1))

input_shape = ( x[1], 1)


# load your trained ANN
from keras.models import load_model
classifier = load_model('classifier.h5')


# Create model without the softmax function (at the output layer)
model_wo_sm = iutils.keras.graph.model_wo_softmax(classifier)

# Analysing ANN prediction w.r.t. the predicted alternative 
LRP_1 = innvestigate.create_analyzer("lrp.epsilon", model_wo_sm)

# Analysing ANN prediction w.r.t. a selected alternative (analyst can select) 
LRP_2 = innvestigate.create_analyzer("lrp.epsilon", model_wo_sm, neuron_selection_mode="index")

# select the observation that you want to inspect using LRP method (say observation number 5)
a = X_all [5,:,:]
a = np.reshape(a,(1,a.shape[0],1))


# Analysing ANN prediction w.r.t. the predicted alternative (alt # 2)
analysis_LRP_1 = LRP_1.analyze(a)

# Analysing ANN prediction w.r.t. a selected alternative (alt # 2 for this example)
analysis_LRP_2 = LRP_2.analyze(a,2)
