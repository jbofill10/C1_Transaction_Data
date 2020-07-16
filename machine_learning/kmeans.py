from sklearn.cluster import KMeans

import pickle
import pandas as pd

def run():

    df = pd.read_pickle('../Data/pickles/clustering_data')

    # print(df)
    #
    # scores = {'clusters': [], 'scores': []}
    #
    # for num_of_clusters in range(1, 31):
    #     scores['clusters'].append(num_of_clusters)
    #     scores['scores'].append(KMeans(n_clusters=num_of_clusters, verbose=1, random_state=0).fit(df).score(df))
    #
    # with open('../Data/pickles/elbow', 'wb') as file:
    #     pickle.dump(scores, file)

    kmeans = KMeans(n_clusters=3, verbose=1, random_state=0)

    cluster_labels = [f'cluster_{i}' for i in kmeans.fit(df).labels_]

    df['cluster'] = cluster_labels

    df.to_pickle('../Data/pickles/clustered_df', protocol=4)

    print(df)

if __name__ == '__main__':
    run()