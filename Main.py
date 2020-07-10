from tqdm import tqdm

import pandas as pd
import os, json

import numpy as np


def main():

    dataset_content = list()
    cols = None

    # Way faster than use pd.concat every line
    # Tricky needing me to sort the dict ;)

    with open('Data/transactions.txt', 'r') as file:
        lines = file.readlines()

        for line in tqdm(lines):
            temp_dict = json.loads(line)
            temp_dict = dict(sorted(temp_dict.items()))

            if cols is None: cols = list(temp_dict.keys())
            values = list(temp_dict.values())
            for i in range(len(values)):

                if isinstance(values[i], str) and len(values[i]) == 0:
                    values[i] = np.nan
                elif isinstance(values[i], str):
                    values[i] = values[i].strip()

            dataset_content.append(values)

    df = pd.DataFrame(dataset_content, columns=cols)

    pd.to_pickle(df, 'Data/pickles/dataset', protocol=4)


if __name__ == '__main__':
    main()