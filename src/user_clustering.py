from scipy.cluster.hierarchy import dendrogram, linkage, ward
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np

def add_clusters_to_frame(or_data, clusters, SLICE_LENGTH):
    #or_frame = pd.DataFrame(data=or_data)
    #or_frame_labelled = pd.concat([or_frame, pd.DataFrame(clusters)], axis=1, join='inner')
    #or_frame_labelled.rename(columns={or_frame_labelled.columns[-1]: "clusters"}, inplace=True)
    #return (or_frame_labelled)

    clusters_array = np.zeros(or_data.shape[0], dtype=int)
    np.put(clusters_array, np.arange(clusters.shape[0]), clusters)
    or_data['clusters'] = clusters_array[:]
    return or_data[:SLICE_LENGTH]

def cluster_users(distance_matrix,dataframe, SLICE_LENGTH):
    
    cluster = AgglomerativeClustering(n_clusters=112, affinity='euclidean', linkage='ward')

    ### The clusters assigned to user proiles
    clusters = cluster.fit_predict(distance_matrix)

    df = add_clusters_to_frame(dataframe, clusters, SLICE_LENGTH)
    return df, clusters
