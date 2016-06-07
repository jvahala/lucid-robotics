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

class Subspace(object):
	def __init__(self,U): 
		self.U = U 				#U is orthogonal subspace ndarray object n by p
		self.n = U.shape[0]		#number of elements in the subspace
		self.p = U.shape[1]		#number of features in subspace

	def projectOnMe(self,X):
		#project a different subspace Y (m by r, m and r possible not equal to n and p) onto the space spaned by self.U
		def extendX(X):
			#inds = np.array([added_ind1, added_ind2, added_ind3, ...]) int between {1,2,...,max_ind-1}
			#X is too small to be projected on U, so need to add additional points
			if len(X) > self.n: #check you didn't use the wrong function (should be done for you already though )
				print 'whoops, extendX() is not for you'
				return
			else: 
				num_add = self.n-len(X)
				print 'adding ', num_add, ' elements'
				interps = np.random.randint(len(X)-1, size=num_add)	#select interpolation indices at the halfway points ]along the elements of the basis {0.5,1.5,2.5...,max_ind-0.5}
				interps = interps.astype('float64') 
				interps += 0.5
				
				interps = np.sort(interps)
				ceil_interps = np.ceil(interps)
				Xnew = np.ones((len(X)+num_add,1))
				for col in X.T: 					#for each column of X, interpolate
					value_add = np.interp(interps,np.arange(len(col)),col)
					col = np.insert(col,ceil_interps,value_add)
					col99 = np.insert(col,ceil_interps,np.ones(len(ceil_interps))*-99) #fills in added entries with -99
					inds = np.where(col99==-99)		#inds which will be removed later
					Xnew = np.hstack((Xnew,col.reshape(len(col),1)))
				X = Xnew[:,1:]	#ignore first column
			return X, inds

		def contractX(X):
			#X is too large to be projected on U, so need to remove points
			#inds = np.array([added_ind1, added_ind2, added_ind3, ...])
			if len(X) < self.n: 
				print 'whoops, contractX() is not for you'
				return
			else: 
				num_remove = len(X) - self.n
				print 'removing ', num_remove, ' elements'
				removes = np.random.choice(len(X)-2,size=num_remove,replace=False)+1	#select from {1,2,...max_ind-1} without replacement
				removes = np.sort(removes)
				inds = np.empty_like(removes)
				for i,r in enumerate(removes): 
					inds[i] = r-1-i 	#index after which to place the new element when adding them back for interpolation
				Xnew = np.ones((len(X)-num_remove,1))
				for col in X.T:
					col = np.delete(col,removes,axis=0)
					Xnew = np.hstack((Xnew,col.reshape(len(col),1)))
				X = Xnew[:,1:]
			return X, inds

		def resolveProjection(Z,inds,status): 
			if status == 0: 
				print 'status is go'
				return Z
			elif status == +1: 
				print 'removing uncessary dumb additions'
				#remove unnecessary added rows from Z 
				Z = np.delete(Z,inds,axis=0)
				return Z
			elif status == -1: 
				#add necessary removed points to Z 
				print 'adding the important addtions back'
				interps = inds + 0.5
				Znew = np.ones((len(Z)+len(inds),1))
				for col in Z.T: 
					values = np.interp(inds+0.5, np.arange(len(col)), col) 
					col = np.insert(col,np.ceil(interps),values)
					Znew = np.hstack((Znew,col.reshape(len(col),1)))
				Z = Znew[:,1:]
				return Z

		status = 0						#default that self.U and X are the same length
		inds = []
		if len(X) < self.n: 
			print 'extending'
			X,inds = extendX(X)
			status = +1					#indices have been added, will need to remove these from the projection later
		elif len(X) > self.n: 
			print 'contracting'
			X,inds = contractX(X)
			status = -1					#indices have been removed, will need to interpolate in projection later
		Z = projectToSubspace(X,self.U)
		Z = resolveProjection(Z,inds,status)
		return Z


def projectToSubspace(X,Y): 
	'''
	Purpose: 
	Embeds a set of features X (in R^(n by k)) onto a reduced dimension subspace Y (in R^(n by r)), r < k, via least squares approximation, Z = Xw where w = inv(X'X)X'Y

	Inputs: 
	X - n by k feature array (ndarray type)
	Y - n by r feature array, (r<k, ndarray type)

	Outputs: 
	Z - n by r ndarray subspace projection of X onto Y

	'''
	w = np.linalg.lstsq(X,Y) 
	Z = X.dot(w[0])
	return Z 

def matchLengths(X,Y):
	'''
	Purpose: 
	Via interpolation or 

	Inputs: 
	X - n by k feature array (ndarray type)
	Y - n by r feature array, (r<k, ndarray type)

	Outputs: 
	Z - n by r ndarray subspace projection of X onto Y

	'''
	return ''

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

X = np.random.randint(5,size=(100,8))
Y = np.random.randint(2,size=(100,2))
Z = projectToSubspace(X,Y)