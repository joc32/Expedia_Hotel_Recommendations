from numpy.linalg import matrix_rank
import torch
import numpy as np

def svd(clusters,sliced_matrix):

    # We need to compress utility matrix by replacing user_id with user_cluster
    # and remove all rows where user_cluster repeats
    unique_clusters = np.unique(clusters, return_index=True)
    # create matrix user_cluster x hotel_cluster
    clustered_matrix = sliced_matrix[unique_clusters[1]]

    #  Find the rank of the cluster Matrix. Will be used for the dimensions of U and V.
    # rank is the number of independent columns of a matrix
    rank=matrix_rank(clustered_matrix)

    # Perfom Stochastic Gradient Descent to find U,V for X=U@V.T
    U,V=sgd(clustered_matrix,rank,1000,0.01,0.01)

    # Performs multiplication of the two matrices  X=U@V.T
    new_clustered=svd(U,V,clustered_matrix)
    return new_clustered


# Implements Stochastic Gradient Descent
# lr: learning rate.
# lamda: normalisation factor to keep the values from getting to high.
# num_epochs, the number of iterations the algorithm will perform.
def sgd(M,rank,num_epochs,lr,lamda):
  # n: number of user_clusters (112)
  n=M.shape[0]
  # k: number of hotel_clusters (100)
  k=M.shape[1]
  # Define U and V
  # U[M,rank] M: number of user clusters
  # V[rank,K] K: number of hotel clusters.
  U=torch.rand(n,rank)
  V=torch.rand(k,rank)

  for epoch in range(num_epochs):
    for r in range(n):
      for c in range(k):
        # We are updating U and V for every known value.
        if M[r][c]>0:
            e1=M[r][c]-V[c,:].t()@U[r,:]
            e2=M[r][c]-U[r,:].t()@V[c,:]
            U[r,:]=U[r,:]+lr*(e1*V[c,:]-lamda*U[r,:])
            V[c,:]=V[c,:]+lr*(e2*U[r,:]-lamda*V[c,:])
  return U,V

# Performs the multiplication of matrices X=U@V.T
def construct_svd(U,V,M):
  n=M.shape[0]
  k=M.shape[1]
  new_M = np.zeros([n,k])
  for r in range(n):
    for c in range(k):
      new_M[r][c]=U[r,:]@V[c,:].T
  return new_M

