import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import os
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
from src.combination import create_destination_matrix, create_r_matrix, recommend_best_hotel_cluster
from src.evaluation import map5eval


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


# Load our train dataset

train = pd.read_csv(os.path.join('datasets','1percent.csv'))
print('number of rows in sample', len(train))

#Create a dataframe only with columns we need. And sort it by user_id.
temp = train[['user_id','hotel_cluster','is_booking', 'srch_destination_id']]
temp.sort_values(by=['user_id'],inplace=True)

# out of 376703 ids, 262231 are unique.
print('\n Number of unique User IDs:',temp['user_id'].nunique())

# Create Rating column for our dataframe.
# Where booking == 1, give rating 5 otherwise give rating 1.
temp['rating'] = np.where((temp['is_booking'] == 1),5,1)


# Create utility matrix out of temp
utility_matrix = um.create_utility_matrix(df=temp)

# Then slice it to 1000 rows for further analysis.
sliced_matrix = utility_matrix[0:1000, :]

#Perform the Cosine Distance Calculation on our sliced matrix.
normalised = um.get_distance_matrix(sliced_matrix)
um.plot_hgram(normalised,'sliced_utility_cosine_normalised.png')

# Eyad code
clusters = cluster_users(normalised,temp)
print(clusters.head(100))

# Clusters variable is created by Eyads code
# Perfom SVD on the clustered matrix to reduce sparcity
svd_matrix=svd.construct_svd(clusters,sliced_matrix)

# Please keep the clusters matrix as its my input. Thanks -Eria :)

# remove is_booking column which is no more needed
clusters = clusters.iloc[:, [0, 1, 3, 4]]

#  sort by srch_destination_in
clusters.sort_values(by=['srch_destination_id'], inplace=True)

destination_matrix = create_destination_matrix(clusters)
r_matrix = create_r_matrix(destination_matrix, sliced_matrix)


# Test our algorithm on one user cluster. Not needed later.
user_cluster = 1
top_5_hotels = recommend_best_hotel_cluster(user_cluster, r_matrix, destination_matrix)
print(top_5_hotels)


clusters['recommended_train'] = pd.Series(index=clusters.index, dtype=object)

# Vectorised implementation doesnt want to work, for now left.
#clusters['recommended_train'] = recommend_best_hotel_cluster(clusters['hotel_cluster'], r_matrix, destination_matrix)

for i,row in clusters.iterrows():
    clusters.at[i,'recommended_train'] = recommend_best_hotel_cluster(row[1], r_matrix, destination_matrix)

print('recommended clusters are')
print(clusters.head(100))


map5eval(clusters['recommended_train'], clusters['hotel_cluster'])


# TEST DATASET FROM HERE
test = pd.read_csv(os.path.join('datasets','test.csv'))
test_ids = test[['user_id']]

process_test(test_ids, train)
