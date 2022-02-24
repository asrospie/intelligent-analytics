import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import keras_tuner as kt
from sklearn import metrics
import datetime
import re
import os
import seaborn as sns
import sys

train_df = pd.read_excel('fonts_training.xlsx')
test_df = pd.read_excel('fonts_test.xlsx')

x_train_full = train_df[['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5', 'Cat 6', 'Cat 7', 'Cat 8',
       'Cat 9', 'Cat 10', 'Cat 11', 'Cat 12', 'Cat 13', 'Cat 14']].to_numpy()
y_train_full = train_df[['A', 'B',
       'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
       'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']].to_numpy()

x_scaler = StandardScaler().fit(x_train_full)
x_sc_train_full = x_scaler.transform(x_train_full)


x_train, x_valid, y_train, y_valid = train_test_split(x_sc_train_full, y_train_full)

def build_model(n_units, n_layers, learning_rate, dropout=False):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(14,), name='input_layer'))
    
    for i in range(n_layers):
        model.add(keras.layers.Dense(units=n_units, activation='relu', name=f'hidden_layer_{i}'))
        
    if dropout:
        model.add(keras.layers.Dropout(rate=.5, name='dropout_layer'))
        
    model.add(keras.layers.Dense(units=26, activation='softmax', name='output_layer'))
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        learning_rate=learning_rate,
        metrics=['accuracy']
    )
    
    return model

def tune_and_build_model(hp):
    n_units = hp.Int('units', min_value=20, max_value=200, step=20)

    n_layers = hp.Int('layers', min_value=1, max_value=5, step=1)

    dropout = hp.Boolean('dropout')

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=.1, sampling='log')

    model = build_model(n_units, n_layers, learning_rate, dropout=dropout)

    return model

CALLBACKS = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)]

tuner = kt.BayesianOptimization(
    hypermodel=tune_and_build_model,
    objective='val_accuracy',
    max_trials=100,
    overwrite=True,
    directory='tuning',
    project_name='fr_bayesian'
)

tuner.search(x_train, y_train, epochs=50, callbacks=CALLBACKS, validation_data=(x_valid, y_valid))