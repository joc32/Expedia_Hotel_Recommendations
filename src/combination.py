import numpy as np


def create_destination_matrix(subset_data):
    """
    Generate destination matrix, which is matrix of srch_destination_id to hotel_cluster;
    iterate over each hotel cluster and calculate rating for each destination of hotel cluster separately

    :param subset_data: subset df of user logs containing srch_destination_id
    :return: destination matrix
    """
    np_subset_matrix = subset_data.to_numpy()
    print(np_subset_matrix[:10])
    last_dest_id = np_subset_matrix[-1, 2]
    n_hotel_clusters = 100

    # empty D matrix: (n of destinations x n of hotel clusters)
    # !!! destination_ids start with index 0 (e.i. dest_id 1 is at position 0) !!!
    destination_matrix = np.zeros([last_dest_id, n_hotel_clusters])

    # calculate part of destination matrix for each hotel cluster separately
    for i in range(n_hotel_clusters):
        # create filtered_matrix only with rows where hotel cluster == i
        indices_to_filter = np.where(np_subset_matrix[:, 1] == i)
        filtered_matrix = np_subset_matrix[indices_to_filter]

        # if there are no values for looped hotel_cluster, then go to next iteration
        if len(filtered_matrix) == 0:
            continue
        # create matrix with ratings only
        rank_matrix = filtered_matrix[:, 3]

        # group by destination and sum all ratings for each destination
        d = np.diff(filtered_matrix[:, -2])
        d = np.where(d)[0]
        indices = np.r_[0, d + 1]
        # summed all ratings for each destination
        summed_rating = np.add.reduceat(rank_matrix, indices, axis=0)

        # get number of ratings for each destination
        np_counts = np.array(np.unique(filtered_matrix[:, 2], return_counts=True))
        # transpose matrix to get values by columns
        # destination_ids of all destinations used
        destination_ids = np_counts.T[:, 0]
        # number of times one hotel_cluster has been rated in one  destination_id
        count_values = np_counts.T[:, 1]

        # calculate average rating for each destination
        average = np.divide(summed_rating, count_values)

        # fill destination matrix with average values
        # shift each destination_id in matrix by one to start from index 0
        destination_matrix[destination_ids - 1, i] = average

    return destination_matrix


def create_r_matrix(utility_matrix, destination_matrix):
    """
    Generate R matrix, which is a dot product of utility matrix and destination matrix;

    :param destination_matrix
    :param utility_matrix
    :return: R matrix
    """
    return utility_matrix.dot(destination_matrix.T)


def recommend_best_hotel_cluster(user_cluster, r_matrix, destination_matrix):
    """
    For given user cluster ID recommend top 5 hotel clusters based on R matrix and destination matrix ratings;

    :param user_cluster
    :param r_matrix
    :param destination_matrix
    :return: top_5_h_clusters
    """

    # find best destination_id for given user_cluster
    sel_row = r_matrix[user_cluster]
    top_destination_id = np.argsort(-sel_row)[0]

    # find best hotel_cluster for given destination_id
    sel_dest_row = destination_matrix[top_destination_id]
    top_5_h_clusters = np.argsort(-sel_dest_row)[:5]

    # print 5 best matches for given user_cluster
    #print('for user_cluster {}, best hotel_clusters are {}'.format(user_cluster, top_5_h_clusters))

    return top_5_h_clusters


def recommend_5_top_hotel_cluster_2(user_cluster, destination_id, utility_matrix, destination_matrix):
    # element-wise multiplication of hotel_clusters for given user_cluster and hotel_cluster for given destination_id
    r_vector = utility_matrix[user_cluster] * destination_matrix[destination_id - 1]
    top_5 = np.argsort(-r_vector)[:5]
    return top_5
