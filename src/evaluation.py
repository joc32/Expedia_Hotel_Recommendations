import numpy as np
import pandas as pd


def map5eval(predicted, actual, k=5):
    y = np.array([np.array(xi) for xi in predicted])
    metric = 0.
    for i in range(5):
        metric += np.sum(actual==y[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@5', metric


def random_recommendations(df):
    df['random'] = pd.Series(index=df.index, dtype=object)

    for i, row in df.iterrows():
        df.at[i, 'random'] = np.random.randint(0,101,5)

    map5_score = map5eval(df['random'].values, df['hotel_cluster'].values)
    print(map5_score)


def append_user_clusters(test_df, clustered_df):
    """
        Add to test dataframe column with user clusters, if given user_id already exists in train dataset,
        then use user cluster from dataset, otherwise let deecision tree to determine the  user cluster.
        :param test_df: test dataframe
        :param clustered_df: train dataframe with already clustered users
        :return test_df with user clusters column.
    """
    # make matrix from user_id and clusters
    user_id_clusters = clustered_df.as_matrix(columns=['user_id', 'clusters'])
    test_df['clusters'] = test_df['user_id'].apply(
        lambda user_id:
        get_cluster(user_id, user_id_clusters) if user_id in user_id_clusters[:, 0]
        else get_decision_tree(user_id))
    return test_df


def get_cluster(user_id, user_id_clusters):
    i = np.where(user_id_clusters[:, 0] == user_id)[0][0]
    return user_id_clusters[i, 1]


def get_decision_tree(user_id):
    #print('NOT IMPLEMENTED')
    return 0
