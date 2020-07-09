from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

import os, pickle


def run(data):
    if not os.path.isfile('Data/pickles/model_results/xgboost'):
        params = {
            'n_estimators': [10, 30, 50, 70, 90, 125, 500],
            'max_depth': [1, 3, 5],
            'learning_rate': [0.001, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.5, 0.7, 0.9],
            'reg_alpha': [0.01, 0.05, 0.07, 0.09, 0.5, 1],
            'reg_lambda': [0.01, 0.05, 0.07, 0.09, 0.5, 1],
            'n_jobs': [-1]
        }

        grid = GridSearchCV(estimator=XGBClassifier(tree_method='gpu_hist'), param_grid=params, cv=5, verbose=3)

        grid.fit(data['x_train'], data['y_train'])

        with open('Data/pickles/model_results/xgboost', 'wb') as file:
            pickle.dump(grid, file)
    else:
        with open('Data/pickles/model_results/xgboost', 'rb') as file:
            grid = pickle.load(file)
    return grid
