from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import os, pickle


def run(data):
    if not os.path.isfile('Data/pickles/model_results/svc'):
        params = {
            'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 70, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['poly', 'rbf'],
            'degree': [1, 2, 3, 4, 5]
        }

        grid = GridSearchCV(estimator=SVC(), param_grid=params, cv=5, verbose=3)

        grid.fit(data['x_train'], data['y_train'])

        with open('Data/pickles/model_results/svc', 'wb') as file:
            pickle.dump(grid, file)
    else:
        with open('Data/pickles/model_results/svc', 'rb') as file:
            grid = pickle.load(file)
    return grid
