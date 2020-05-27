import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import warnings
import os
import itertools
import sys
import time

import src.utility_matrix as um
import src.user_clustering as clt
import src.svd as svd
import src.combination as comb
import src.evaluation as evaluate
import resource

matplotlib.use('Agg')

# Pandas options
pd.set_option('display.precision', 10)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
warnings.filterwarnings('ignore')



def train(slice):

    # Load train dataset only with columns we need.
    temp = pd.read_csv(
        os.path.join('datasets', 'train.csv'),
        index_col=False,
        usecols=['user_id', 'hotel_cluster', 'is_booking', 'srch_destination_id'])

    n = int(temp.shape[0] * slice)
    print(' Doing on slice ', n)

    print(' Number of rows in sample', len(temp))

    # Reorder columns to desired order
    temp = temp.reindex(columns=['user_id', 'hotel_cluster', 'is_booking', 'srch_destination_id'])

    # sort columns in dataframe by user_id.
    temp.sort_values(by=['user_id'], inplace=True)

    # out of 376703 ids, 262231 are unique.
    print('  Number of unique User IDs in train dataset:', temp['user_id'].nunique())

    # Create Rating column for our dataframe.
    # Where booking == 1, give rating 5 otherwise give rating 1.
    temp['rating'] = np.where((temp['is_booking'] == 1), 5, 1)

    print('\n  Create new Sample  ')

    # Create a sample to minimize the size of the utility matrix
    sliced_temp, c = np.split(temp, [int(n)])
    print('  New Slice size: ' + str(sliced_temp.shape))

    print('  max RAM usage (linux kb, mac bytes)',
          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    print('  Number of unique User IDs in  split dataset:', sliced_temp['user_id'].nunique())

    print('  Utility Matrix creation.  ')
    # Create utility matrix out of temp
    utility_matrix = um.create_utility_matrix(df=sliced_temp)

    # Then slice it for further analysis.

    #um.plot_hgram(sliced_matrix,'first_slice_train'+str(SLICE_LENGTH)+'.png')
    #sliced_matrix = utility_matrix[0:SLICE_LENGTH, :]
    #um.plot_hgram(sliced_matrix,'second')


    print('  Utility Matrix Normalisation.  ')
    # Perform the Cosine Distance Calculation on our sliced matrix.
    normalised = um.get_distance_matrix(utility_matrix)

    print('  Matrix Clustering.  ')
    clustered_df, clusters = clt.cluster_users(normalised, sliced_temp)

    print('  Performing SVD  ')
    # Perform SVD on the clustered matrix to reduce sparsity
    utility_svd_matrix = svd.construct_svd(clusters, utility_matrix)
    #um.plot_hgram(utility_svd_matrix, 'Clustered_UM_after_SVD'+str(SLICE_LENGTH)+'.png')

    # remove is_booking column which is no more needed
    temp = temp.loc[:, ['user_id', 'hotel_cluster', 'srch_destination_id', 'rating']]

    #  sort by srch_destination_in
    temp.sort_values(by=['srch_destination_id'], inplace=True)

    print('  Destination Matrix creation.  ')
    destination_matrix = comb.create_destination_matrix(temp)

    return clustered_df, destination_matrix, utility_svd_matrix


def predict_train(clustered_df, destination_matrix, utility_svd_matrix):

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

    plt.figure()
    plt.title('Variance in Recommended Clusters')
    plt.hist(list(itertools.chain(*clustered_df['recommended_train'].values)), bins=100)
    plt.savefig('figures/hotel_cluster_counts.png')

    map5_score = evaluate.map5eval(clustered_df['recommended_train'].values, clustered_df['hotel_cluster'].values)
    print('map5', map5_score)
    return map5_score


def predict_test(clustered_df, destination_matrix, utility_svd_matrix):
    # open test dataset with required columns
    test_df = pd.read_csv(
        os.path.join('datasets', 'test.csv'),
        index_col=False,
        usecols=['id', 'user_id', 'srch_destination_id'])
    test_df = test_df.reindex(columns=['id', 'user_id', 'srch_destination_id'])

    print('\n  Clustering user_ids from test dataset.  ')
    # append user clusters to test dataset
    test_df = evaluate.append_user_clusters(test_df, clustered_df)

    print('  Calculating recommendations and saving them to datasets/test_results.txt.  ')
    # save results to test_results.txt file
    test_results = open(os.path.join('datasets', 'test_results.txt'), 'w')
    test_results.write('id,hotel_cluster\n')

    for i, row in test_df.iterrows():
        top5_results = comb.recommend_5_top_hotel_clusters(test_df.at[i, 'clusters'], test_df.at[i, 'srch_destination_id'], utility_svd_matrix, destination_matrix)
        test_results.write(str(row['id']) + ',' + ' '.join(str(r) for r in top5_results) + '\n')
    test_results.close()


def main():

    start = time.time()

    if len(sys.argv) > 1:
        slice = sys.argv[1]
    else:
        # Specify dataset slice percentage on which make the recommendation
        slice = 0.005

    print('training recommendation algorithm on', slice * 100, '% of dataset')
    clustered_df, destination_matrix, utility_svd_matrix = train(slice=slice)

    print('running predictions on ', slice * 100, '% of  dataset')
    #map5_score = predict_train(clustered_df, destination_matrix, utility_svd_matrix)

    predict_test(clustered_df, destination_matrix, utility_svd_matrix)

    print('it took', time.time() - start, 'seconds to run.')


if __name__ == '__main__':
    main()
