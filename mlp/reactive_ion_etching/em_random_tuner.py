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

FINAL_MODEL_NAME = sys.argv[1]

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


def build_model(n_units, n_layers, learning_rate, dropout, activation='relu'):
	model = keras.models.Sequential()

	# Input Layer
	model.add(keras.layers.InputLayer(input_shape=(6,), name='input_layer'))

	# Hidden Layers
	for i in range(n_layers):
		model.add(keras.layers.Dense(units=n_units, activation=activation, name=f'hidden_layer_{i}'))

	if dropout:
		model.add(keras.layers.Dropout(rate=.50))

	# Output Layer
	model.add(keras.layers.Dense(units=4, name='output_layer'))
	
	model.compile(
		optimizer='adam',
		learning_rate=learning_rate,
		loss='mean_squared_error',
		metrics=['mean_squared_error']
	)

	return model

def tune_and_build_model(hp):
	n_units = hp.Int('units', min_value=32, max_value=512, step=32)

	n_layers = hp.Int('layers', min_value=1, max_value=5, step=1)

	learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=.1, sampling='log')

	dropout = hp.Boolean('dropout')

	model = build_model(n_units, n_layers, learning_rate, dropout)

	return model

# Tuner with random search
tuner = kt.RandomSearch(
	hypermodel=tune_and_build_model,
	objective='mean_squared_error',
	max_trials=50,
	executions_per_trial=5,
	overwrite=True,
	directory='tuning',
	project_name='etch_testing'
)

tuner.search(x_scaled_full, y_scaled_full, epochs=30)

print(tuner.results_summary())

# Get best 5 models
best_models = tuner.get_best_models(num_models=5)

# Test best models with cross validation
best_avg_mses = np.array([])
for idx, m in enumerate(best_models):
    loo = LeaveOneOut()
    avg_mse = 0
    for train, valid in loo.split(x_scaled_full):
        x_train, x_valid = x_scaled_full[train], x_scaled_full[valid]
        y_train, y_valid = y_scaled_full[train], y_scaled_full[valid]

        m.fit(x_train, y_train, epochs=30, verbose=0)
        avg_mse += metrics.mean_squared_error(m.predict(x_valid), y_valid)

    avg_mse /= (len(x_scaled_full))
    best_avg_mses = np.append(best_avg_mses, avg_mse)
    print(f'Model # {idx} :: Avg MSE={avg_mse:.5f}')
print(best_avg_mses)


best_model_idx = np.argmin(best_avg_mses)
print(f'Best Model Index: {best_model_idx} with an average validation MSE of {best_avg_mses[best_model_idx]}')

# build final model
best_hp = tuner.get_best_hyperparameters(num_trials=5)[best_model_idx]
model = tune_and_build_model(best_hp)

model.fit(x_scaled_full, y_scaled_full, epochs=30)

timestamp = re.sub(r'[-\s+:\.]', '_', str(datetime.datetime.now()))
timestamp = timestamp[:len(timestamp) - 10]
model.save(f'{FINAL_MODEL_NAME}_{timestamp}')