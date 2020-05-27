from sklearn.cluster import AgglomerativeClustering


def add_clusters_to_frame(or_data, clusters):
    or_data['clusters'] = or_data['user_id'].apply(lambda user_id: clusters[user_id])
    return or_data


def cluster_users(distance_matrix, dataframe):

    cluster = AgglomerativeClustering(n_clusters=112, affinity='precomputed', linkage='average')
    # The clusters assigned to user profiles
    clusters = cluster.fit_predict(distance_matrix)

    df = add_clusters_to_frame(dataframe, clusters)
    return df, clusters
