from numpy.linalg import matrix_rank
import torch
import numpy as np


def construct_svd(clusters, sliced_matrix):
    """
    Performs the Singular Value Decomposition on the utility matrix, to fill in the unknown values.
    It seeks to find a low-rank matrix X=UV.T, that minimizes the sum-square
    distance to the fully observed target matrix M.
    U[N,rank]
    V[rank,K]
     where N=the number of user clusters
           K=the number of hotel clusters
           rank the number of independent columns in M.
    It uses Stochastic Gradient Descent to minimize the sum-square distance to M.
    :param clusters: Is the clusters of users created by the Hierarchical Clustering algorithm.
    :param sliced_matrix: Is the utility matrix, sliced, due to RAM limitations.
    :return low- rank matrix X.
    """
    # We need to compress utility matrix by replacing user_id with user_cluster
    # and remove all rows where user_cluster repeats
    unique_clusters = np.unique(clusters['clusters'], return_index=True)
    # create matrix user_cluster x hotel_cluster
    clustered_matrix = sliced_matrix[unique_clusters[1]]

    #  Find the rank of the cluster Matrix. Will be used for the dimensions of U and V.
    # rank is the number of independent columns of a matrix
    rank = matrix_rank(clustered_matrix)

    # Perfom Stochastic Gradient Descent to find U,V for X=U@V.T
    U, V = sgd(clustered_matrix, rank, 1000, 0.01, 0.01)

    # Performs multiplication of the two matrices  X=U@V.T
    new_clustered = svd(U, V, clustered_matrix)
    return new_clustered


def sgd(m, rank, num_epochs, a, lamda):
    """
     Implements the Strochastic Gradient Descent.
     Tries to find U and V that minimizes the sum square distance to M.
     This is done through derivatives of the sum square distance to M,
     with respect to U and V.
     U and V are initialised to random values drawn from Gaussian distribution.
     For the number of epochs:
        the algorithm iterates over all known values of the M, and calculates the
        new values of U and V based on the derivatives as shown below:
        Ui := Ui + α((Mik − VkT Ui)Vk − λUi)
        Vk := Vk + α((Mik − UiT Vk)Ui − λVk)

    :param m: utility matrix
    :param rank: the rank of the utility matrix M.
    :param num_epochs: number of iterations
    :param a: the learning rate
    :param lamda: regularization parameter
    :return: U,V
    """
    # n: number of user_clusters (112)
    n = m.shape[0]
    # k: number of hotel_clusters (100)
    k = m.shape[1]
    # Define U and V
    # U[N,rank] N: number of user clusters
    # V[rank,K] K: number of hotel clusters
    U = torch.rand(n, rank)
    V = torch.rand(k, rank)
    for epoch in range(num_epochs):
        for r in range(n):
            for c in range(k):
            # We are updating U and V for every known value.
                if m[r][c] > 0:
                    e1 = m[r][c]-V[c, :].t()@U[r, :]
                    e2 = m[r][c]-U[r, :].t()@V[c, :]
                    U[r, :] = U[r, :]+a*(e1*V[c, :]-lamda*U[r, :])
                    V[c, :] = V[c, :]+a*(e2*U[r, :]-lamda*V[c, :])
    return U, V


def svd(u, v, m):
    """
    Calculates matrix X based on:
        X=U@V.T

    :param u: U
    :param v: V
    :param m: utility matrix
    :return: X
    """
    n = m.shape[0]
    k = m.shape[1]
    new_M = np.zeros([n, k])
    for r in range(n):
        for c in range(k):
            # X=U@V.T
            new_M[r][c] = u[r, :] @ v[c, :].T
    return new_M
