from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import pickle


def run(data):
    params = {
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [1, 3, 5, 7],
        'max_leaf_nodes': [None, 3, 5, 7]
    }

    grid = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, cv=5, verbose=3)

    grid.fit(data['x_train'], data['y_train'])

    with open('Data/pickles/model_results/dec_tree', 'wb') as file:
        pickle.dump(grid, file)
