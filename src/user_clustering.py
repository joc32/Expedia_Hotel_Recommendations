from scipy.cluster.hierarchy import dendrogram, linkage, ward
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from kmeans_pytorch import kmeans
import torch
from matplotlib import pyplot as plt


def add_clusters_to_frame(or_data, clusters, SLICE_LENGTH):
    #or_frame = pd.DataFrame(data=or_data)
    #or_frame_labelled = pd.concat([or_frame, pd.DataFrame(clusters)], axis=1, join='inner')
    #or_frame_labelled.rename(columns={or_frame_labelled.columns[-1]: "clusters"}, inplace=True)
    #return (or_frame_labelled)

    or_data['clusters'] = or_data['user_id'].apply(lambda user_id: clusters[user_id])
    return or_data

    #to_array = np.zeros(clusters.shape, dtype=int)
    #or_data = clusters[or_data['user_id']]
    #np.put(to_array, clusters, clusters)
    #or_data["clusters"]

    #clusters_array = np.zeros(or_data.shape[0], dtype=int)
    #np.put(clusters_array, np.arange(clusters.shape[0]), clusters)
    #or_data['clusters'] = clusters_array[:]
    #return or_data[:SLICE_LENGTH]

def cluster_users(distance_matrix, dataframe, SLICE_LENGTH, utility_matrix):

    #cluster = AgglomerativeClustering(n_clusters=112, affinity='precomputed', linkage='average')
    ### The clusters assigned to user proiles
    #clusters = cluster.fit_predict(distance_matrix)

    x = torch.from_numpy(utility_matrix)
    clusters, centers = kmeans(X=x, num_clusters=112, distance='cosine')
    clusters = clusters.numpy()

    df = add_clusters_to_frame(dataframe, clusters, SLICE_LENGTH)
    return df, clusters
