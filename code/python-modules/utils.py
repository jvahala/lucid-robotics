'''
module: utils.py 
use: contains functions associated with general functionality that are not unique to any particular part of the project
'''

import numpy as np
from scipy.stats import mode #this isnt actually used i think

def getDifferenceArray(vector): 
	'''
	Purpose: 
	Takes an m by n vector and returns a symmetric array with elements representing the different between components in the vector
	array[i,j] = ||vector[i,:] - vector[j,:]||^2

	Inputs: 
	vector - m by n ndarray type representing a set of joint positions, for example 

	Outputs: 
	array - n by n ndarray with the i-jth element equal to the norm^2 difference between the ith and jth rows of vector

	'''
	vec_len = len(vector)
	array = np.zeros((vec_len,vec_len))
	for i in range(0,vec_len):
		for j in range(i,vec_len):
			array[i,j] = np.linalg.norm((vector[i,:]-vector[j,:]))
	array = symmetrize(array)
	return array

def getSimilarityArray(feature_array,similarity_method = 'exp',k_nn = 5):
	'''
	Purpose: 
	Computes the similarity array for a given feature set, similarity method, and k_nearest_neighbors value
	Part of the spectral clustering process

	Inputs: 
	feature_array - set of features 
	similarity_method - method to use for computing the similarity array: 
		--'exp' computes W[i,j] = exp(-||xi - xj||^2 / 2)
		--'norm' computes W[i,j] = ||xi - xj||^2
		--'chain' is specifically for the 'chain' generateData type
	k_nn - number of nearest neighbors to consider (k_nn=5 means only the top 5 largest similarity values are kept nonzero)

	Outputs: 
	sim_array - symmetric array of similarity strength values

	'''

	allowed_methods = ['exp','norm','chain']
	if similarity_method not in allowed_methods:
		print 'ERROR: Not a valid similarity_method'
		return 
	else:
		sim_array = np.zeros((len(feature_array),len(feature_array)))
		i = 0
		j = 0
		for rowi in feature_array:
			for rowj in feature_array: 
				if i <= j: 
					difference = (rowi-rowj).T
					if similarity_method == 'exp':
						sim_array[i,j] = np.exp(-1*((difference.T).dot(difference)))
					elif similarity_method == 'norm':
						sim_array[i,j] = difference.T.dot(difference)
					elif similarity_method == 'chain':
						if np.linalg.norm(difference) <= 1.5: 
							if ((i != int(len(feature_array)/2.)-1) and (j != int(len(feature_array)/2.))):
								sim_array[i,j] = 1
							if i == j: 
								sim_array[i,j] = 1
				j += 1
			i += 1
			j = 0
		sim_array = sim_array - np.diag(sim_array.diagonal()) #remove diagonal nonzero values
		if k_nn != -1: 
			for rowi in sim_array:
				ind = np.argpartition(rowi, -1*k_nn)[(-1*k_nn):]

				for i in range(len(rowi)):
					if i not in ind: 
						rowi[i] = 0; 
		return symmetrize(sim_array)

def symmetrize(array): 
	'''
	Purpose: 
	Returns the symmetric version of an upper or lower triangular array

	Inputs: 
	array - upper OR lower triangular ndarray 

	Outputs: 
	symmetric version of array

	'''
	return array + array.T - np.diag(array.diagonal())

def normalize(array,normalizer):
	'''
	Purpose: 
	Normalize an array by some 'normalizer' value

	Inputs: 
	array - an ndarray type
	normalizer - non int-type value 

	Outputs: 
	array - output of (array/normalizer)

	'''
	array = (1.0/normalizer)*array
	return array

def runningAvg(vector,N): 
	'''
	Purpose: 
	Performs a runningAvg calculation on a 1d array 'vector' and averages over N spaces 

	Inputs: 
	vector - ndarray 1-dimensional array 
	N - number of elements to average over 

	Outputs: 
	vector with each element being the runningAvg over N elements - same size as original vector

	'''
	return np.convolve(vector, np.ones(N,)/(N*1.0))[(N-1):]

def orderStates(vector): 
	'''
	Purpose: 
	Orders states so that first defined state is a 0, second defined state is a 1, etc

	Inputs: 
	vector - 1 dimensional array of a relatively small number of ints 

	Outputs: 
	ordered_vector - vector of same size as original vector but with the first few states ordered 

	'''
	order_hold = []
	for ind,elt in enumerate(vector): 
		if ind == 0: 
			order_hold.append(elt)
			ordered_vector = [0]
		else:
			if elt not in order_hold:
				order_hold.append(elt)
			ordered_vector.append(order_hold.index(elt))

	return ordered_vector

def generateData(N,form='bull',dim=2):
    '''
	Purpose: 
	Generates (N by dim) ndarray of a type described by 'form'
	Particularly useful for testing clustering methods 

	Inputs: 
	N - length of data set
	dim - number of dimensions in dataset (ie dim = 2) 
	form - data set type
		--'sep' compiles a dataset with two distinct groups 
		--'bull' compiles a dataset of a bullseye shape (one labeled group within a ring of the other group)
		--'chain' compiles a dataset of a linear chain with a label break in between them

	Outputs: 
	X - compiled data array of 'form' type
	y - labels associated with each of the N examples of X 

	'''

    X = np.zeros((N,dim),dtype = np.float16)
    y = np.zeros((N,1), dtype = np.int_)
    
    if form == 'sep': #seperate clusters of data
        base1 = np.ones((1,dim))
        base2 = np.zeros((1,dim))
        cnt = 0
        while cnt < np.floor(N/2): 
            X[cnt,:] = base1 + 0.5*(np.random.rand(1,dim)*2.0-1.)
            y[cnt] = 1
            cnt += 1
        while cnt < N:
            X[cnt,:] = base2 + 0.5*(np.random.rand(1,dim)*2.0-1.)
            y[cnt] = -1
            cnt += 1
        y.shape = (N,)
        return X,y

    elif form == 'bull': #inner cluster surrounded by ring of points
        cnt=0;
        X = np.zeros((N,dim),dtype = np.float16)
        y = np.zeros((N,1), dtype = np.int_)
        totalg1 = 0
        totalg2 = 0
        while cnt < N :
            x = 2*np.random.rand(1,dim)-1;
            if np.linalg.norm(x) < 0.15 and totalg1<=(N-np.floor(N/1.2)):
                X[cnt,:] = x;
                y[cnt] = +1
                cnt=cnt+1;
                totalg1 +=1
            elif (np.linalg.norm(x) > 0.5 and np.linalg.norm(x) < 0.55) and totalg2<(N-(N-np.floor(N/1.2))):
                X[cnt,:] = x;
                y[cnt] = -1
                cnt=cnt+1;
                totalg2 += 1
        y.shape = (N,)
        return X,y

    elif form == 'chain': #linear chain graph of N points
    	X = np.zeros((N,dim),dtype = np.float16)
    	for i in np.arange(N):
    		X[i,:] = i
    		if i < N/2.:
    			y[i] = +1
    		else: 
    			y[i] = -1
    	y.shape = (N,)
       	return X,y

