from tensorflow import keras
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import pickle
import skopt
import sys
from skopt.plots import plot_convergence

AE_LAYERS = int(sys.argv[1])


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

N_CLASSES = 10

x_train = x_train / 255.0
x_test = x_test / 255.0

# Preprocess data for cnn
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

def build_model(n_units=[50], n_layers=1, activation='relu', optimizer='adam', learning_rate=1e-3, ae_layers=1):
    layer_map = { 1: 'one', 2: 'two', 3: 'three', 4: 'four' }
    ae_loaded = keras.models.load_model(f'./models/ae_{layer_map[ae_layers]}_layers/')

    # Freeze all AE layer weights
    for l in ae_loaded.layers:
        l.trainable = False

    # Transfer only the encoder layers
    model = keras.models.Sequential(ae_loaded.layers[:-ae_layers])

    # Add new layers
    for i in range(n_layers):
        model.add(keras.layers.Dense(
            units=n_units[i],
            activation=activation,
            name=f'hidden_layer_{i}'
        ))

    # output layer
    model.add(keras.layers.Dense(10, activation='softmax', name='output_layer'))

    opt = None
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate)

    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create search space for optimizer
MIN_FIRST_LAYER = int(784 / (2**AE_LAYERS) * 0.75)
MAX_FIRST_LAYER = int(784 / (2**AE_LAYERS))
MIN_SECOND_LAYER = 12
MAX_SECOND_LAYER = int(784 / (2**AE_LAYERS) * 0.75)
SPACE = [
    skopt.space.Integer(MIN_FIRST_LAYER, MAX_FIRST_LAYER, name='n_units_one'),
    skopt.space.Integer(MIN_SECOND_LAYER, MAX_SECOND_LAYER, name='n_units_two'),
    skopt.space.Integer(1, 2, name='n_layers'),
    skopt.space.Real(1e-4, .1, prior='log-uniform', name='learning_rate'),
    skopt.space.Categorical(['relu', 'selu', 'tanh'], name='activation'),
    skopt.space.Categorical(['adam', 'sgd'], name='optimizer'),
]

N_CALLS = 10

model_calls = [ i for i in range(N_CALLS) ]
model_dict = {}

EARLY_STOPPING = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

@skopt.utils.use_named_args(SPACE)
def objective(**params):
    model_num = model_calls.pop(0)
    print('###############################################################################')
    
    cv = KFold(5)
    
    path = f'./test_models/ae_mlp_{AE_LAYERS}_checkpoint_{model_num}.h5'
    checkpoint = keras.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=path,
        verbose=0,
        save_weights_only=True,
        save_best_only=True
    )
    scores = []
    count = 0
    for tr_idx, vl_idx in cv.split(x_train, y_train):
        x_tr, x_vl = x_train[tr_idx], x_train[vl_idx]
        y_tr, y_vl = y_train[tr_idx], y_train[vl_idx]

        n_units = []
        if params['n_layers'] == 2:
            n_units = [params['n_units_one'], params['n_units_two']] 
        else:
            n_units = [params['n_units_one']]
        
        ts_model = build_model(
            n_units=n_units,
            n_layers=params['n_layers'],
            learning_rate=params['learning_rate'],
            activation=params['activation'],
            optimizer=params['optimizer'],
            ae_layers=AE_LAYERS
        )
        if count == 0:
            print(ts_model.summary())
        
        print(f'Training model {model_num}.{count}...')
        ts_model.fit(x_tr, y_tr, validation_data=(x_vl, y_vl), callbacks=[EARLY_STOPPING, checkpoint], epochs=30, verbose=0)
        print(f'Finished Training... {5-count-1} models left.')
        count += 1
        
        ts_predictions = ts_model.predict(x_vl)
        class_predictions = [ np.argmax(x) for x in ts_predictions ]
        score = accuracy_score(class_predictions, y_vl)
        scores.append(score)
        
    print(f'--------------------------------------------------')
    print(f'Run #{model_num}')
    print('n_units_one : {}'.format(params['n_units_one']))
    print('n_units_two : {}'.format(params['n_units_two']))
    print('learning_rate : {}'.format(params['learning_rate']))
    print('activation : {}'.format(params['activation']))
    print('optimizer : {}'.format(params['optimizer']))
    print(f'Avg. Val. Acc. :: {np.mean(scores)}')
    
    model_dict[model_num] = {
        'n_units': [params['n_units_one'], params['n_units_two']],
        'n_layers': params['n_layers'],
        'learning_rate': params['learning_rate'],
        'activation': params['activation'],
        'optimizer': params['optimizer'],
        'path': path,
        'avg_score': np.mean(score)
    }
    
    print('###############################################################################')
    return -np.mean(scores)

opt_results = skopt.gp_minimize(objective, SPACE, n_calls=N_CALLS, random_state=0)
skopt.dump(opt_results, f'./tuning/ae_mlp_{AE_LAYERS}_opt_results.pkl')

with open(f'./tuning/ae_mlp_{AE_LAYERS}_model_dict.pickle', 'wb') as handle:
    pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Find best values
avg_scores = [ (i, val['avg_score']) for i, val in model_dict.items() ]
max_val = 0
max_idx = 0
for i in avg_scores:
    if i[1] > max_val:
        max_val = i[1]
        max_idx = i[0]

bp = model_dict[max_idx]

print(bp)

# Plot convergence graph
ax = plot_convergence(opt_results)
plt.savefig(f'./plots/ae_mlp_{AE_LAYERS}_convergence.png')
plt.clf()

# Build final model from best results
final_model = build_model(
    n_units=bp['n_units'],
    n_layers=bp['n_layers'],
    activation=bp['activation'],
    optimizer=bp['optimizer'],
    learning_rate=bp['learning_rate'],
    ae_layers=AE_LAYERS   
)
final_model.build()
final_model.load_weights(bp['path'])
final_model.save(f'./models/ae_mlp_{AE_LAYERS}/')

predictions = final_model.predict(x_test)
class_predictions = [ np.argmax(x) for x in predictions ]
print(classification_report(y_test, class_predictions))

from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, class_predictions)
ax = sns.heatmap(cm.T, square=True, annot=True, fmt='d')
ax.set(title='Autoencoder MLP', xlabel='Predicted', ylabel='True')
plt.savefig(f'./plots/ae_mlp_{AE_LAYERS}_cm.png')