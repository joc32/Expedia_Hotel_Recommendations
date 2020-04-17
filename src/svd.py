from numpy.linalg import matrix_rank
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from src.utility_matrix import plot_hgram


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
    unique_clusters = np.unique(clusters, return_index=True)
    # create matrix user_cluster x hotel_cluster
    clustered_matrix = sliced_matrix[unique_clusters[1]]

    #  Find the rank of the cluster Matrix. Will be used for the dimensions of U and V.
    # rank is the number of independent columns of a matrix
    rank = matrix_rank(clustered_matrix)

    # Perfom Stochastic Gradient Descent to find U,V for X=U@V.T
    epochs = 200

    plot_hgram(clustered_matrix,'Clustered_UM_before_SVD')
    U, V = sgd(clustered_matrix, rank=rank, num_epochs=epochs, a=0.01, lamda=0.01, calculate_loss='FALSE')

    new_clustered = U@V.t()

    return new_clustered.numpy()


def sgd(m, rank, num_epochs, a, lamda, calculate_loss):
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

    losses = []
    m = torch.from_numpy(m) #Cast the numpy matrix to Torch Tensor.

    n = m.shape[0]  # n: number of user_clusters (112)
    k = m.shape[1]  # k: number of hotel_clusters (100)
    # Define U and V
    U = torch.rand(n, rank)  # U[N,rank] N: number of user clusters
    V = torch.rand(k, rank)  # V[K,rank] K: number of hotel clusters
    for epoch in range(num_epochs):
        for r in range(n):
            for c in range(k):
            # We are updating U and V for every known value.
                if m[r][c] > 0:
                    e1 = m[r][c]-V[c, :].t()@U[r, :]
                    e2 = m[r][c]-U[r, :].t()@V[c, :]
                    U[r, :] = U[r, :]+a*(e1*V[c, :]-lamda*U[r, :])
                    V[c, :] = V[c, :]+a*(e2*U[r, :]-lamda*V[c, :])
        if calculate_loss == 'TRUE':
            print(epoch)
            losses.append(torch.nn.functional.mse_loss(input=m, target=U @ V.t(), reduction='sum'))

    if calculate_loss == 'TRUE':
        plt.figure()
        plt.ticklabel_format(style='plain')
        plt.title('SVD LOSS epoch num %s rank %s' % (num_epochs, rank))
        plt.plot(range(len(losses)), losses)
        plt.savefig('figures/svd_plot_' + str(num_epochs) + 'epochs.png')

    return U, V


def sgd_jozef(m, rank, num_epochs, a, lamda, calculate_loss):
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

    losses = []
    binary_mask = np.ma.make_mask(m).astype(int)
    m = torch.from_numpy(m) #Cast the numpy matrix to Torch Tensor.

    n = m.shape[0]  # n: number of user_clusters (112)
    k = m.shape[1]  # k: number of hotel_clusters (100)
    # Define U and V
    U = torch.rand(n, rank)  # U[N,rank] N: number of user clusters
    V = torch.rand(k, rank)  # V[K,rank] K: number of hotel clusters
    for epoch in range(num_epochs):
        for r in range(n):
            for c in range(k):
                if m[r][c] > 0:
                    e = binary_mask[r][c] - U[r, :] @ V[c, :].t()
                    U[r, :] = U[r, :] + a * e * V[c, :]
                    V[c, :] = V[c, :] + a * e * U[r, :]
        if calculate_loss == 'TRUE':
            print(epoch)
            criterion = torch.nn.MSELoss()
            loss = torch.sqrt(criterion(m, U@V.t()))
            losses.append(loss)

    if calculate_loss == 'TRUE':
        plt.figure()
        plt.ticklabel_format(style='plain')
        plt.title('SVD LOSS epoch num %s rank %s' % (num_epochs, rank))
        plt.plot(range(len(losses)), losses)
        plt.savefig('figures/svd_plot_' + str(num_epochs) + 'epochs.png')

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
