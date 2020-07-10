import pickle
from machine_learning import decision_tree, logistic_reg, random_forest, svm, xgboost


def train():

    with open('Data/pickles/preproc_data', 'rb') as file:
        data = pickle.load(file)

    results = {
        'dec_tree': decision_tree.run(data),
        'log_reg': logistic_reg.run(data),
        'rf': random_forest.run(data),
        'svm': svm.run(data),
        'xgboost': xgboost.run(data),
    }


    print(results)



if __name__ == '__main__':
    train()