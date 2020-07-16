import pandas as pd
import numpy as np


def main():

    df = pd.read_json('Data/transactions.txt', lines=True)
    df.replace('', np.nan, inplace=True)
    print(df.isnull().sum())
    pd.to_pickle(df, 'Data/pickles/dataset', protocol=4)


if __name__ == '__main__':
    main()