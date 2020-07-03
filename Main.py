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
    df_in_4 = len(df)//4

    # must be my way so I can use pickles with git

    df_1 = df.iloc[0:df_in_4]
    df_2 = df.iloc[df_in_4:df_in_4*2]
    df_3 = df.iloc[df_in_4*2:df_in_4*3]
    df_4 = df.iloc[df_in_4*3:df_in_4*4 + (len(df) - df_in_4*4)]

    pd.to_pickle(df_1, 'Data/pickles/full_data/df_1')
    pd.to_pickle(df_2, 'Data/pickles/full_data/df_2')
    pd.to_pickle(df_3, 'Data/pickles/full_data/df_3')
    pd.to_pickle(df_4, 'Data/pickles/full_data/df_4')


if __name__ == '__main__':
    main()