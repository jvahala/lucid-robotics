'''
module: clusters.py 
use: contains functions associated clustering / unsupervised learning 
'''

import numpy as np 
from kmeans import kplusplus
from utils import getSimilarityArray

def getDegreeArray(sim_array): #convert array W into respective Degree array, Dii = sum(i=1 to n) Wij
	'''
	Purpose: 
	Computes the Degree array 'D' in the spectral clustering process from the similarity array
	Dii = \sum_{i=1}^n Wij, ie the sum of each row of the similarity array

	Inputs: 
	sim_array - Similarity array Wij retrieved from getSimilarityArray()

	Outputs: 
	D - degree array (described in Purpose

	'''
	D = np.zeros((sim_array.shape[0],sim_array.shape[0]))
	for i in range(0,sim_array.shape[0]):
		D[i,i] = np.sum(sim_array[i,:])
	return D

def getLaplacian(W,D): 
	'''
	Purpose: 
	Returns the Laplacian of the similarity array W and the degree array D 
	For use with spectral clustering

	Inputs: 
	W - similarity array from getSimilarityArray()
	D - degree array from getDegreeArray

	Outputs: 
	L = D-W, the laplacian 

	'''
	return D-W

def spectralClustering(features,similarity_method='exp',k_nn=5,basis_dim=2,num_clusters=2): 
	'''
	Purpose: 
	Performs spectral clustering into 'num_clusters' clusters on data defined in the ndarray 'features'

	Inputs: 
	features - n examples by k features ndarray (n>k preferred)
	similarity_method - method to use for computing the similarity array: 
		--'exp' computes W[i,j] = exp(-||xi - xj||^2 / 2)
		--'norm' computes W[i,j] = ||xi - xj||^2
		--'chain' is specifically for the 'chain' generateData type
	k_nn - number of nearest neighbors to consider in similarity array
	basis_dim - number of svd basis vectors to consider for input to kmeans++ algorithm
	num_clusters - number of clusters for kmeans++ to sort the data into

	Outputs: 
	labels - 1 by n array of assigned cluster labels for each feature example
	centers - cluster centers array (basis_dim by num_clusters) representing each of the k cluster centers 

	'''

	W = getSimilarityArray(features,similarity_method,k_nn)
	D = getDegreeArray(W)
	L = getLaplacian(W,D)
	U,s,V = np.linalg.svd(L,full_matrices=0)
	U = U[:,-1*basis_dim:]
	labels, centers = kplusplus(U.T,num_clusters)
	return labels, centers, U 
