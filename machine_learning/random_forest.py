from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import os, pickle


def run(data):
    if not os.path.isfile('Data/pickles/model_results/rf'):
        params = {
            'n_estimators': [10, 50, 100, 300, 500],
            'max_depth': [1, 3, 5],
            'bootstrap': [True, False],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'oob_score': [True, False]
        }

        grid = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, cv=5, verbose=3)

        grid.fit(data['x_train'], data['y_train'])

        with open('Data/pickles/model_results/rf', 'wb') as file:
            pickle.dump(grid, file)
    else:
        with open('Data/pickles/model_results/rf', 'rb') as file:
            grid = pickle.load(file)
    return grid
