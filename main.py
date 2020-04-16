import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import itertools
import sys
import time

# Our imports
import src.utility_matrix as um
import src.user_clustering as clt
import src.svd as svd
import src.combination as comb
import src.decision_tree as tree
import src.evaluation as evaluate

# Pandas options
pd.set_option('display.precision', 10)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
warnings.filterwarnings('ignore')



def run(slice):
    SLICE_LENGTH = int(slice)

    # Load train dataset only with columns we need.
    temp = pd.read_csv(
        os.path.join('datasets', '1percent.csv'),
        index_col=False,
        usecols=['user_id', 'hotel_cluster', 'is_booking', 'srch_destination_id'])
    print('  number of rows in sample', len(temp))

    # Reorder columns to desired order
    temp = temp.reindex(columns=['user_id', 'hotel_cluster', 'is_booking', 'srch_destination_id'])

    # sort columns in dataframe by user_id.
    temp.sort_values(by=['user_id'], inplace=True)

    # out of 376703 ids, 262231 are unique.
    print('\n  Number of unique User IDs:', temp['user_id'].nunique())


    # Create Rating column for our dataframe.
    # Where booking == 1, give rating 5 otherwise give rating 1.
    temp['rating'] = np.where((temp['is_booking'] == 1), 5, 1)

    print('  Utility Matrix creation.  ')
    # Create utility matrix out of temp
    utility_matrix = um.create_utility_matrix(df=temp)
    # Then slice it for further analysis.

    n = SLICE_LENGTH  # for 2 random indices
    index = np.random.choice(utility_matrix.shape[0], n, replace=False)
    sliced_matrix = utility_matrix[index, :]
    #um.plot_hgram(sliced_matrix,'first_slice_train'+str(SLICE_LENGTH)+'.png')
    #sliced_matrix = utility_matrix[0:SLICE_LENGTH, :]
    #um.plot_hgram(sliced_matrix,'second')


    print('  Utility Matrix Normalisation.  ')
    # Perform the Cosine Distance Calculation on our sliced matrix.
    normalised = um.get_distance_matrix(sliced_matrix)

    print('  Matrix Clustering.  ')
    clustered_df, clusters = clt.cluster_users(normalised, temp, SLICE_LENGTH)

    print('  Performing SVD  ')
    # Perform SVD on the clustered matrix to reduce sparsity
    utility_svd_matrix = svd.construct_svd(clusters, sliced_matrix)
    #um.plot_hgram(utility_svd_matrix, 'Clustered_UM_after_SVD'+str(SLICE_LENGTH)+'.png')

    # remove is_booking column which is no more needed
    temp = temp.loc[:, ['user_id', 'hotel_cluster', 'srch_destination_id', 'rating']]

    #  sort by srch_destination_in
    temp.sort_values(by=['srch_destination_id'], inplace=True)

    print('  Destination Matrix creation.  ')
    destination_matrix = comb.create_destination_matrix(temp)

    # Vectorised implementation doesnt want to work, for now left.
    #clusters['recommended_train'] = recommend_best_hotel_cluster(clusters['hotel_cluster'], r_matrix, destination_matrix)
    #clustered_df['recommended_train'] = recommend_5_top_hotel_cluster_2(clustered_df['clusters'],clustered_df['srch_destination_id'],utility_matrix,destination_matrix)

    print('\n Calculating Recommendations...  \n')

    print('hotel cluster value counts')
    print(clustered_df['clusters'].value_counts())

    plt.figure()
    plt.hist(clustered_df['clusters'],bins=len(clustered_df['clusters'].value_counts()))
    plt.savefig('figures/user_cluster_distribution.png')
    clustered_df['recommended_train'] = pd.Series(index=clustered_df.index, dtype=object)

    k=0
    for i, row in clustered_df.iterrows():
        clustered_df.at[i, 'recommended_train'] = comb.recommend_5_top_hotel_clusters(clustered_df.at[i, 'clusters'], clustered_df.at[i, 'srch_destination_id'], utility_svd_matrix, destination_matrix)
        #k+=1
        #print(k, 'iteration out of ',SLICE_LENGTH, clustered_df.at[i,'clusters'],clustered_df.at[i,'srch_destination_id'])

    plt.figure()
    plt.title('Variance in Recommended Clusters')
    plt.hist(list(itertools.chain(*clustered_df['recommended_train'].values)), bins=100)
    plt.savefig('figures/hotel_cluster_counts.png')

    map5_score = evaluate.map5eval(clustered_df['recommended_train'].values, clustered_df['hotel_cluster'].values)
    print('map5', map5_score)


    # TEST DATASET FROM HERE
    #test = pd.read_csv(os.path.join('datasets', 'test.csv'))
    #test_ids = test[['user_id']]
    #evaluate.rocess_test(test_ids, train)

if __name__ == '__main__':

    start = time.time()

    if len(sys.argv) > 1:
        sl = sys.argv[1]
    else:
        sl = 1000

    print('running recommendation algorithm with slice of', sl)
    run(slice=sl)
    print('it took', time.time()-start, 'seconds to run.')
