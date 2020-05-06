import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import spatial
from sklearn.metrics import pairwise_distances
import seaborn as sns
import os
from scipy.spatial.distance import cdist


def create_utility_matrix(df):
    """
    Generate utility matrix from dataframe;
    first create blank X users and Y hotel clusters matrix,
    and fill it with ratings from dataframe df.

    :param df: Dataframe from which you want to create UM.
    :return: utility matrix
    """

    # Create an int Numpy matrix out of 0th, 1th, and 4th columns from dataframe.
    data_matrix = df.iloc[:, [0, 1, 4]].to_numpy().astype(int)

    # Get the last user id so we know how big our utility matrix will be.
    last_user_id = int(df.iloc[-1].user_id)  # 1198777

    utility_matrix = np.zeros(shape=(last_user_id + 1, 100))
    print('  our utility matrix has shape', utility_matrix.shape)

    x = [data_matrix[:,0], data_matrix[:,1]]
    np.add.at(utility_matrix, x, data_matrix[:,2])
    return utility_matrix


def get_distance_matrix(matrix):
    """
    :param matrix: utility matrix
    :return: distance matrix with cosine distances
    """

    # Calculations done on sliced utility matrix

    # create column vector which has row sums of ratings.
    row_sums = np.sum(matrix,axis=1)

    #create intermediate column vector with rating sum / N clusters
    inter_mediate = row_sums / matrix.shape[1]

    # Take away the rating_sum / N clusters column vector
    # from the sliced matrix
    matrix = matrix - inter_mediate[:,None]

    # Generate distance matrix using Sklearn library out of
    # The normalised matrix.

    dist_out = cdist(matrix, matrix, metric='cosine')
    #dist_out = 1-pairwise_distances(matrix, metric="cosine")
    return dist_out

def test_cosine():
    # Calculations done on the array in the excel spreadsheet.
    test_arr = np.array([[0, 1, 1, 5, 1, 0], [1, 1, 1, 5, 1, 0]])
    n_clusters = 6
    print(test_arr)  # Our excel data

    # Column vector with row sums of ratings.
    row_sums = np.sum(test_arr, axis=1)
    print(row_sums)

    # Column vector with row sums / N clusters
    inter_mediate = row_sums / len(test_arr[0])
    print(inter_mediate)

    # Subtraction
    # Subtract the column vector with row sums / N clusters from
    # Our ratings matrix.
    test_arr = test_arr - inter_mediate[:, None]
    print(test_arr)

    # Sum of squares
    ssq = np.sum(test_arr ** 2, axis=1)
    print(ssq)

    # Squared sum of squares
    ssq = np.sqrt(ssq)
    print(ssq)

    # last step. A_k * B_k / squared sum of squares A, ssq B.
    cd = np.sum(np.prod(test_arr, axis=0)) / np.prod(ssq)
    print('our cosine similarty is', cd)

    # Test it using Library
    result = 1 - spatial.distance.cosine(test_arr[0], test_arr[1])
    assert cd == result

    print('distance matrix')
    dist_out = 1 - pairwise_distances(test_arr, metric="cosine")
    print(dist_out)


def calculate_ram(matrix):
    print('slice shape is', matrix.shape[0], matrix.shape[1])
    z = (matrix.size * matrix.itemsize)
    print('Matrix takes: ', z/1000000000,'GB RAM')


def plot_hgram(matrix,name):
    plt.figure(figsize=(20,20))
    #sns.set(font_scale=4.5)
    sns.heatmap(matrix,cmap='jet')
    plt.title(name, fontsize=30)
    plt.savefig(os.path.join('figures',name))