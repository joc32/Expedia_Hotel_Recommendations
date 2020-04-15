import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
import itertools

pd.set_option('display.precision', 10)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
import src.utility_matrix as um
import src.user_clustering as clt
import src.svd as svd
import src.combination as comb
import src.decision_tree as tree
from src.user_clustering import  cluster_users
from src.combination import create_destination_matrix, create_r_matrix, recommend_best_hotel_cluster, recommend_5_top_hotel_cluster_2
from src.evaluation import map5eval

preds = np.array([[3,2,4,5,6], [5,3,1,2,4]])
label = np.array([[10], [30]])

smth = map5eval(preds, label)

print(smth)


def get_user_cluster():
    print('NOT IMPLEMENTED')
    exit()

def get_decision_tree():
    print('NOT IMPLEMENTED')
    exit()

def process_test(test_ids, train):
    test_ids['present_in_train'] = test_ids['user_id'].isin(train['user_id'])
    print(test_ids[test_ids['present_in_train'] == True])
    test_ids['recommendations'] = np.where((test_ids['present_in_train'] == True), get_user_cluster(), get_decision_tree())


# SLICE_LENGTH = 1000
#
# # Load our train dataset
#
train = pd.read_csv(os.path.join('datasets','1percent.csv'))
# print('number of rows in sample', len(train))
#
# #Create a dataframe only with columns we need. And sort it by user_id.
# temp = train[['user_id','hotel_cluster','is_booking', 'srch_destination_id']]
# temp.sort_values(by=['user_id'],inplace=True)
#
# # out of 376703 ids, 262231 are unique.
# print('\n Number of unique User IDs:',temp['user_id'].nunique())
#
# # Create Rating column for our dataframe.
# # Where booking == 1, give rating 5 otherwise give rating 1.
# temp['rating'] = np.where((temp['is_booking'] == 1),5,1)
#
#
# # Create utility matrix out of temp
# utility_matrix = um.create_utility_matrix(df=temp)
#
# # Then slice it to 1000 rows for further analysis.
# sliced_matrix = utility_matrix[0:SLICE_LENGTH, :]
#
# #Perform the Cosine Distance Calculation on our sliced matrix.
# normalised = um.get_distance_matrix(sliced_matrix)
# um.plot_hgram(normalised,'sliced_utility_cosine_normalised'+str(SLICE_LENGTH)+'.png')
#
# # Eyad code
# clustered_df, clusters = cluster_users(normalised,temp)
# print(clustered_df.head())
#
# # Clusters variable is created by Eyads code
# # Perform SVD on the clustered matrix to reduce sparsity
# utility_svd_matrix = svd.construct_svd(clusters,sliced_matrix)
#
# # Please keep the clusters matrix as its my input. Thanks -Eria :)
#
# # remove is_booking column which is no more needed
# temp = temp.iloc[:, [0, 1, 3, 4]]
#
# #  sort by srch_destination_in
# temp.sort_values(by=['srch_destination_id'], inplace=True)
#
# destination_matrix = create_destination_matrix(temp)
# r_matrix = create_r_matrix(utility_svd_matrix, destination_matrix)
#
#
# # Test our algorithm on one user cluster. Not needed later.
# user_cluster = 1
# destination_id = 0
# top_5_hotels = recommend_5_top_hotel_cluster_2(user_cluster, destination_id, utility_svd_matrix, destination_matrix)
# print(top_5_hotels)
#
# #top_5_hotels = recommend_best_hotel_cluster(user_cluster, r_matrix, destination_matrix)
# #print(top_5_hotels)
#
# user_cluster = 2
# destination_id = 4
# top_5_hotels = recommend_5_top_hotel_cluster_2(user_cluster, destination_id, utility_svd_matrix, destination_matrix)
# print(top_5_hotels)
#
# #top_5_hotels = recommend_best_hotel_cluster(user_cluster, r_matrix, destination_matrix)
# #print(top_5_hotels)
#
#
# clustered_df['recommended_train'] = pd.Series(index=clustered_df.index, dtype=object)
#
# # Vectorised implementation doesnt want to work, for now left.
# #clusters['recommended_train'] = recommend_best_hotel_cluster(clusters['hotel_cluster'], r_matrix, destination_matrix)
# #clustered_df['recommended_train'] = recommend_5_top_hotel_cluster_2(clustered_df['clusters'],clustered_df['srch_destination_id'],utility_matrix,destination_matrix)
#
#
# for i, row in clustered_df.iterrows():
#     clustered_df.at[i, 'recommended_train'] = recommend_5_top_hotel_cluster_2(clustered_df.at[i, 'clusters'], clustered_df.at[i, 'srch_destination_id'], utility_svd_matrix, destination_matrix)
#
# print('recommended clusters are')
# print(clustered_df.head(100))

clustered_df = pd.read_pickle('pickled_1000.pkl')

plt.figure()
plt.title('Variance in Recommended Clusters')
plt.hist(list(itertools.chain(*clustered_df['recommended_train'].values)),bins=100)
plt.savefig('figures/hotel_cluster_counts.png')

map5neco = map5eval(clustered_df['recommended_train'], clustered_df['hotel_cluster'])
print('map5', map5neco)





# TEST DATASET FROM HERE
test = pd.read_csv(os.path.join('datasets', 'test.csv'))
test_ids = test[['user_id']]

process_test(test_ids, train)
