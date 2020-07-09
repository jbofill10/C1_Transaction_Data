from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

import os, pickle


def run(data):
    if not os.path.isfile('Data/pickles/model_results/log_reg'):
        params = {
            'penalty': ['l2', 'none'],
            'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 10, 100],
            'fit_intercept': [True, False],
            'n_jobs': [-1]
        }

        grid = GridSearchCV(estimator=LogisticRegression(max_iter=1000), param_grid=params, cv=5, verbose=3)

        grid.fit(data['x_train'], data['y_train'])

        with open('Data/pickles/model_results/log_reg', 'wb') as file:
            pickle.dump(grid, file)
    else:
        with open('Data/pickles/model_results/log_reg', 'rb') as file:
            grid = pickle.load(file)
    return grid
