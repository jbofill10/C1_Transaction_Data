import pickle
from machine_learning import decision_tree

def train():
    with open('Data/pickles/preprocessed_data/preproc_data') as file:
        data = pickle.load(file)

    decision_tree.run(data)