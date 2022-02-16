import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import LeaveOneOut
import numpy as np
import pandas as pd
import keras_tuner as kt
from sklearn import metrics


data = pd.read_excel('./etching.xlsx')

x_full_df = data[['Pressure', 'RF Power', 'Electrode Gap', 'CCL_4 Flow', 'HE Flow', 'O2 Flow']].astype(float)
y_full_df = data[['Etch Rate -Rp A/min', 'Etch Uniformity', 'Oxide Selectivity - Sox', 'Photoresist Selectivity - Sph']].astype(float)

x_full = x_full_df.to_numpy()
y_full = y_full_df.to_numpy()


# Transform the data
x_scaler = MaxAbsScaler().fit(x_full)
y_scaler = MaxAbsScaler().fit(y_full)

x_scaled_full = x_scaler.transform(x_full)
y_scaled_full = y_scaler.transform(y_full)


def build_model(n_units, n_layers, activation, learning_rate):
	model = keras.models.Sequential()

	# Input Layer
	model.add(keras.layers.Dense(units=6, name='input_layer'))

	# Hidden Layers
	for i in range(n_layers):
		model.add(keras.layers.Dense(units=n_units, activation=activation, name=f'hidden_layer_{i}'))

	# Output Layer
	model.add(keras.layers.Dense(units=4, name='output_layer'))

	model.compile(
		optimizer='adam',
		learning_rate=learning_rate,
		loss='mean_squared_error',
		metrics=['val_mean_squared_error']
	)

	return model

def build_and_tune(hp):
	n_units = hp.Int('units', min_value=32, max_value=512, step=32)

	activation = hp.Choice('activation', ['relu', 'tanh'])

	n_layers = hp.Int('layers', min_value=1, max_value=5, step=1)

	learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=.1, sampling='log')

	model = build_model(n_units, n_layers, activation, learning_rate)

	return model

# Tuner with random search
tuner = kt.RandomSearch(
	hypermodel=build_and_tune,
	objective='mean_squared_error',
	max_trials=50,
	executions_per_trial=5,
	overwrite=True,
	directory='tuning',
	project_name='etch'
)

tuner.search(x_scaled_full, y_scaled_full)

print(tuner.results_summary())

# Get best 5 models
best_models = tuner.get_best_models(num_models=5)

# Cross validate top models to find the best model
for m in best_models:
	loo = LeaveOneOut()
	avg_mse = 0
	for train, valid in loo.split(x_scaled_full):
		m.fit(train)