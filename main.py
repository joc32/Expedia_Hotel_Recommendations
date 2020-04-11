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


# Load our dataset
train = pd.read_csv(os.path.join('datasets','1percent.csv'))
print('number of rows in sample', len(train))
print(train.head(5))

# Create a dataframe only with columns we need. And sort it by user_id.
temp = train[['user_id','hotel_cluster','is_booking', 'srch_destination_id']]
temp.sort_values(by=['user_id'],inplace=True)

# out of 376703 ids, 262231 are unique.
print('\n Number of unique User IDs:',temp['user_id'].nunique())

# Create Rating column for our dataframe.
# Where booking == 1, give rating 5 otherwise give rating 1.
temp['rating'] = np.where((temp['is_booking'] == 1),5,1).astype(float)


# Create utility matrix out of temp
utility_matrix = um.create_utility_matrix(df=temp)

# Then slice it to 1000 rows for further analysis.
sliced_matrix = utility_matrix[0:1000, :]

#Perform the Cosine Distance Calculation on our sliced matrix.
normalised = um.get_distance_matrix(sliced_matrix)
um.plot_hgram(normalised,'sliced_utility_cosine_normalised.png')

# Eyad code



# Please keep the clusters matrix as its my input. Thanks -Eria :)



# Clusters variable is created by Eyads code
# Perfom SVD on the clustered matrix to reduce sparcity
# svd_matrix=svd.svd(clusters,sliced_matrix)