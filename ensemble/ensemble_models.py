from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb

# Data pre-processing
df = pd.read_csv('./water_potability.csv')
print(df.shape)
print(df.isna().sum())

# Fill NA Data
df = df.fillna(df.mean())

x = df.drop(['Potability'], axis=1).to_numpy()
y = df[['Potability']].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.75)

def save_cm(path, pred, actual, title='Confusion Matrix'):
    cm = confusion_matrix(actual, pred)
    ax = sns.heatmap(cm.T, square=True, annot=True, fmt='d')
    ax.set(title=title, xlabel='Predicted', ylabel='Actual')
    plt.savefig(path)
    plt.clf()


def build_and_tune_decision_tree():
    scaler = StandardScaler()
    tree = DecisionTreeClassifier()

    pipe = Pipeline(steps=[('scaler', scaler), ('tree', tree)])

    param_grid = {
        'tree__criterion': ['gini', 'entropy'],
        'tree__splitter': ['best', 'random'],
        'tree__max_depth': [2, 3, 4, 5, 6, 7, 8],
        'tree__max_features': ['auto', 'sqrt', 'log2'],
        'tree__min_samples_split': [2, 3, 4]
    }

    clf = RandomizedSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)

    predicted = clf.predict(x_test)

    # Best Results
    print(f'Best Validation Score {clf.best_score_}')
    print(f'Best Parameters: \n{clf.best_params_}')
    print()
    print(classification_report(predicted, y_test))

    save_cm('./plots/decistion_tree_cm.png', predicted, y_test, title='Decision Tree Confusion Matrix') 

def build_and_tune_xgboost():
    scaler = StandardScaler()
    bst = xgb.XGBClassifier()
    pipe = Pipeline(steps=[('scaler', scaler), ('bst', bst)])

    param_grid = {
        'bst__n_estimators': np.arange(80, 200),
        'bst__max_depth': [2, 3, 4, 5, 6, 7, 8],
        'bst__booster': ['gbtree', 'gblinear', 'dart'],
        'bst__grow_policty': [0, 1]
    }

    clf = RandomizedSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)
    
    predicted = clf.predict(x_test)
    
    # Best Results
    print(f'Best Validation Score {clf.best_score_}')
    print(f'Best Parameters: \n{clf.best_params_}')
    print()
    print(classification_report(clf.predict(x_test), y_test))

    save_cm('./plots/xgboost_cm.png', predicted, y_test, title='XGBoost Confusion Matrix') 

def build_and_tune_bagging():
    scaler = StandardScaler()
    bag = BaggingClassifier()
    pipe = Pipeline(steps=[('scaler', scaler), ('bag', bag)])

    param_grid = {
        'bag__n_estimators': np.arange(8, 50),
        'bag__base_estimator': [DecisionTreeClassifier(), SVC()]
    }

    clf = RandomizedSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)
    
    predicted = clf.predict(x_test)
    
    # Best Results
    print(f'Best Validation Score {clf.best_score_}')
    print(f'Best Parameters: \n{clf.best_params_}')
    print()
    print(classification_report(clf.predict(x_test), y_test))

    save_cm('./plots/bagging_cm.png', predicted, y_test, title='Bagging Confusion Matrix')

def build_and_tune_random_forest():
    scaler = StandardScaler()
    rf = RandomForestClassifier()
    pipe = Pipeline(steps=[('scaler', scaler), ('rf', rf)])

    param_grid = {
        'rf__n_estimators': np.arange(80, 200),
        'rf__criterion': ['gini', 'entropy'],
        'rf__max_depth': np.arange(2, 8),
        'rf__max_features': ['auto', 'sqrt', 'log2']
    }

    clf = RandomizedSearchCV(pipe, param_grid, cv=10, n_jobs=-1)
    clf.fit(x_train, y_train)
    
    predicted = clf.predict(x_test)
    
    # Best Results
    print(f'Best Validation Score {clf.best_score_}')
    print(f'Best Parameters: \n{clf.best_params_}')
    print()
    print(classification_report(clf.predict(x_test), y_test))

    save_cm('./plots/random_forest_cm.png', predicted, y_test, title='Random Forest Confusion Matrix')

print('#################################################################')
print('DECISION TREE CLASSIFIER\n')
build_and_tune_decision_tree()
print('#################################################################')
print('#################################################################')
print('XGBOOST CLASSIFIER\n')
build_and_tune_xgboost()
print('#################################################################')
print('#################################################################')
print('BAGGING CLASSIFIER\n')
build_and_tune_bagging()
print('#################################################################')
print('#################################################################')
print('RANDOM FOREST CLASSIFIER\n')
build_and_tune_random_forest()
print('#################################################################')
