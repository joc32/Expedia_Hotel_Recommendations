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
    # create np matrix with user_id and clusters columns, find rows with unique user ids
    user_id_clusters = clustered_df.as_matrix(columns=['user_id', 'clusters'])
    unique_user_id = np.unique(user_id_clusters[:, 0], return_index=True)

    # create mapping unique user_ids to clusters
    user_id_cluster_matrix = user_id_clusters[unique_user_id[1]][:, 1]
    # create mask where user_id from test exists also in train
    user_id_mask = np.isin(test_df['user_id'], user_id_clusters[:, 0])

    # create new user_ids array where only users which exists
    # TODO implement when user_id does not exist
    masked_user_ids = np.where(user_id_mask == True, test_df['user_id'], 0)

    # lambda function to add user cluster based on user_id
    find_cluster = lambda user_id: user_id_cluster_matrix[user_id]
    find_cluster_f = np.vectorize(find_cluster)
    clusters = find_cluster_f(masked_user_ids)
    test_df['clusters'] = clusters

    return test_df


def get_cluster(user_id, user_id_clusters):
    i = np.where(user_id_clusters[:, 0] == user_id)[0][0]
    return user_id_clusters[i, 1]


def get_decision_tree(user_id):
    #print('NOT IMPLEMENTED')
    return 0
