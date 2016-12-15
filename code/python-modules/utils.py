'''
module: utils.py 
use: contains functions associated with general functionality that are not unique to any particular part of the project
'''

import numpy as np
#from scipy.stats import mode #this isnt actually used i think

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

def numOutsideBounds(_input,bounds): 
	'''
	Purpose: 
	given an input vector of length n and bounds = [lower,upper] each of length n (for each element in the input vector), return the number of elements of the input that are not within the lower and upper bounds

	Inputs: 
	_input - n-length ndarray
	bounds - list of [lower,upper] where lower and upper are each n-length ndarray objects representing the lower and upper bounds that the input should satisfy

	Outputs: 
	num_outside_bounds - integer number of elements of the _input that fell outside of the bounds
	'''
	num_below_lower_bound = np.sum(_input<bounds[0])
	num_above_upper_bound = np.sum(_input>bounds[1])
	num_outside_bounds = num_below_lower_bound+num_above_upper_bound
	return num_outside_bounds

def getBackwardsUniqueOrder(iterable,backward=True): 
	'''
	Purpose: 
	Returns the unique 'most recently seen' order of iterables. For example if the iterable is [0,0,1,3,2,0,1,2,2], this function will return [2,1,0,3]. 

	Inputs: 
	iterable - list or 1D-ndarray with potentially repeated values
	backward - if set to True, then this will return the unique values starting at index 0 of the iterable instead of index -1

	Outputs: 
	reverse - list object 
	'''
	if backward: 
		_iterable = iterable[::-1]
	else:
		_iterable = iterable 
	reverse = [y for ind,y in enumerate(_iterable) if y not in _iterable[0:ind]]
	return reverse

def softmax(x,alpha=-1.0,rescale_=False): 
	if rescale_: 
		x_ = rescale(x)
	else: 
		x_ = x
	expx = np.exp(alpha*np.array(x_))		#take exponential of the -x(i) values in x 
	total = np.sum(expx)	#for use in the denominator
	return expx/float(total)

def rescale(x,max_=10):
	x_scaled = [k/400*float(max_) for k in x]
	return x_scaled

def gaussiansMeet(mu1, std1, mu2, std2): 
	'''
	Purpose: 
	Calculates the intersection points of two gaussian distributions

	Inputs: 
	mu1, mu2 - mean values of the respective guassian distributions
	std1, std2 - standard deviation values of the respective gaussian distributions 

	Outputs: 
	roots - all real values of intersection points 

	'''
	#print 'mu stuff: ', mu1, std1, mu2,std2
	a = 1/(2.*std1**2) - 1/(2.*std2**2)
	b = mu2/(1.*std2**2) - mu1/(1.*std1**2)
	c = mu1**2 /(2.*std1**2) - mu2**2 / (2.*std2**2) - np.log(std2/(1.*std1))
	#print a,b,c
	return np.roots([a,b,c])

class Subspace(object):
	'''
	Purpose: 
	Subspace class that allows for easy projections into the subspace. Used to allow a new set of points that you know would lie in the same subspace (or a similar one) if the number of points/features in the new set of points were the same as the subspace. Thus this is a useful class if there is a structure to be exploited. 

	Functions: 
	self.projectOnMe(self,X) - allows for a differently shaped matrix X (m by r) to be projected on the same space as the base subspace self.U (n by p) if they share a similar structure but for some reason are not the same number of points. 

	callables: 
	self.U - n by p basis matrix for the subspace 
	self.n - number of elements in the subspace
	self.p - number of features in the subspace 

	'''
	def __init__(self,U): 
		'''
		Initialize the subspace class with the basis array U (n by p)
		'''
		self.U = U 				#U is orthogonal subspace ndarray object n by p
		self.n = U.shape[0]		#number of elements in the subspace
		self.p = U.shape[1]		#number of features in subspace

	def projectOnMe(self,X,onlyshape=False):
		'''
		Purpose: 
		Adds or subtracts random points from the matrix X to coincide with the same number of points as self.n. This function uses interpolation between points randomly chosen to add new points to coincide with the dimension of the basis array self.U. 

		Inputs: 
		X - m by r array with m and r possibly different from self.n and self.p 

		Outputs: 
		Z - m by self.p array in the proper subspace self.U 

		'''
		#project a different subspace Y (m by r, m and r possible not equal to n and p) onto the space spaned by self.U
		def extendX(X):
			'''
			Purpose: 
			Adds the necessary number of points to X to match self.n

			Inputs: 
			X - m by r array with m and r possibly different from self.n and self.p 

			Outputs: 
			X - an updated version of X that is now self.n by r 
			inds - array of indices that were added to X to be removed later 

			'''
			#inds = np.array([added_ind1, added_ind2, added_ind3, ...]) int between {1,2,...,max_ind-1}
			#X is too small to be projected on U, so need to add additional points
			if len(X) > self.n: #check you didn't use the wrong function (should be done for you already though )
				#print 'whoops, extendX() is not for you'
				return
			else: 
				num_add = self.n-len(X)
				#print 'adding ', num_add, ' elements'
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
			'''
			Purpose: 
			Removes the necessary number of points to X to match self.n

			Inputs: 
			X - m by r array with m and r possibly different from self.n and self.p 

			Outputs: 
			X - an updated version of X that is now self.n by r 
			inds - array of indices that were removed to X to be added back in later through interpolation in the new basis

			'''
			#X is too large to be projected on U, so need to remove points
			#inds = np.array([added_ind1, added_ind2, added_ind3, ...])
			if len(X) < self.n: 
				#print 'whoops, contractX() is not for you'
				return
			else: 
				num_remove = len(X) - self.n
				#print 'removing ', num_remove, ' elements'
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
			'''
			Purpose: 
			Resolves the projection process after the newly shaped array has been projected on the new subspace by replacing the proper indicies or removing the added indicies placed in inds. 

			Inputs: 
			Z - self.n by self.p array coming from utils.projectToSubspace() 
			inds - indicies of removed or added points in order to shape the projected subspace into the self.U basis. 
			status - (0 = no changes necessary),(+1 = need to remove the unnecessary points that had been added previously),(-1 = need to add in points through interpolation at the appropriate indicies)

			Outputs: 
			Z - m by self.p array in the proper subspace self.U 

			'''
			if status == 0: 
				#print 'status is go'
				return Z
			elif status == +1: 
				#print 'removing uncessary dumb additions'
				#remove unnecessary added rows from Z 
				Z = np.delete(Z,inds,axis=0)
				return Z
			elif status == -1: 
				#add necessary removed points to Z 
				#print 'adding the important addtions back'
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
			#print 'extending'
			X,inds = extendX(X)
			status = +1					#indices have been added, will need to remove these from the projection later
		elif len(X) > self.n: 
			#print 'contracting'
			X,inds = contractX(X)
			status = -1					#indices have been removed, will need to interpolate in projection later
		if onlyshape: 
			return X
		else: 
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


def loader(handover_starts,data_object,n): 
	'''
	Purpose: 
	utility function to get a list with the appropriate start and end frame numbers for a certain data_object and handover_starts. Returns starts = [init_frame,end_frame]

	Inputs: 
	handover_starts - list of the frame numbers for all starts of handovers or general tasks
	data_object - kinectData object that has been filled with data from a file 
	n - the handover number you would like to get the begining and end frames of

	Outputs: 
	starts - list of [init_frame for handover n, end_frame for handover n]

	'''
	try: 
		starts = [handover_starts[n],handover_starts[n+1]]
	except IndexError: 
		starts = [handover_starts[n],data_object.num_vectors-1]
	return starts

def runTasks(handover_starts,data_obj,task_obj,n,max_ind=10): 
	'''
	Purpose: 
	Performs the task class update step for n randomly chosen tasks from the potential dataset of tasks in handover_starts. In other words, for n = 5, 5 different values from handover_starts will be chosen to be used for the task update on the task_obj using data found in data_obj. max_ind represents the total number of handover options in handover_starts to choose from. 

	Inputs: 
	handover_starts - list of the frame numbers for all starts of handovers or general tasks
	data_obj - kinectData object 
	task_obj - process.Task() object 
	n - number of handovers/full tasks to randomly choose
	max_ind - total number of handovers available to be chosen from 

	Outputs: 
	no return object, but task_obj is updated with the new state values and historical data 

	'''
	inds = np.random.randint(max_ind,size=n)
	for i in inds: 
		task_obj.update(data_obj,loader(handover_starts,data_obj,i))

def euclideanDist(point1,point2): 
	'''
	Purpose: 
	Calculates euclidean distance between two points

	Inputs: 
	point1,point2 - same dimensioned points in some space 

	Outputs: 
	output - euclidean distance between the two points, ||point1-point2||_2

	'''
	return np.linalg.norm(point1-point2)

def majorityVote(values): 
	'''
	Purpose: 
	Outputs the most often seen values from a 1d list/array of values along with a list of the sorted indices and sorted values by majority vote 

	Inputs: 
	values - 1d list/array of values that include redundant values 

	Outputs: Two outputs - output1,output2
	output1 - the most often counted value that was found in values 
	output2 - list with two components = [sorted unique values from least often to most often, counts corresponding to the unique values]

	'''
	'''test code (place on own as main)
	x1 = [2]*5+[1]*10+[0]*3			# expected list return- [0,2,1], [3,5,10]
	x2 = [1]*5+[2]*10+[0]*3			# [0,1,2], [3,5,10]
	x3 = [0]*5+[1]*10+[2]*3			# [2,0,1], [3,5,10]
	x4 = [0]*5+[2]*10+[1]*3			# [1,0,2], [3,5,10]
	x5 = [2]*5+[0]*10+[1]*3			# [1,2,0], [3,5,10]
	x6 = [2]*10+[1]*5				# [1,2], [5,10]

	def dothings(x): 
		print x 
		best_val, obj = majorityVote(x)
		print 'most often: ', best_val 
		print 'sorted indicies: ', obj[0]
		print 'sorted count values for indicies: ', obj[1]

	dothings(x1)
	dothings(x2)
	dothings(x3)
	dothings(x4)
	dothings(x5)
	dothings(x6)
	'''
	#print 'np.unique(values): ', np.unique(values), values
	if isinstance(values,list): 
		uValues = np.unique(values).tolist()
		uCounts = [np.sum(np.array(values) == uv) for uv in uValues]
		sorted_inds = np.argsort(uCounts)
		best_val = uValues[sorted_inds[-1]]
		sorted_vals = [int(uValues[x]) for x in sorted_inds]
		sorted_cnts = np.sort(uCounts)
	else: 
		best_val = values 
		sorted_vals = values
		sorted_cnts = len(values)
	return best_val, [sorted_vals, sorted_cnts]

def kNN(new_point, history_points, history_labels, k=5): 
	'''
	Purpose: 
	performs k nearest neighbors algorithm using euclidean distances. Need to give the new point and the past labeled points and labels along with the number of past points to choose the new point label from. 

	Inputs: 
	new_point - 1 by p array representing the new p-featured point in space 
	history_points - n by p array representing the known labeled points in space 
	history-labels - length n list of labels corresponding to the n history_points examples 
	k - nearest neighbors to consider for choosing new point label. The majority vote label from the k closest points to the new point will be output as the new label.

	Outputs: Two outputs in a single list object - [vote,counts_info]
	vote - majority vote label from the k closest points to the new point 
	counts_info - two element list [sorted_inds,counts], sorted_inds: unique labels found in the majority vote search of the k closest elements sorted from fewest examples to most examples, counts: counts of each unique label (same order as in sorted_inds) which in total sum up to k. 

	'''
	distances = []
	for old_point in history_points: 
		distances.append(euclideanDist(new_point,old_point))
	sorted_inds = np.argsort(distances)
	consider_labels = np.array(history_labels)[sorted_inds[0:k]].tolist()
	vote, counts_info = majorityVote(consider_labels)
	return [vote,counts_info]

def compareTaskDef(task_obj,new_labels,kinectData_obj): 
	import process
	new_path = task_obj.definePath(new_labels)
	dummy_task = process.Task(kinectData_obj)	#create dummy task object to printed out the task definition
	dummy_task.path = new_path[0]
	dummy_task.times = new_path[1]
	print 'Expected path (', sum(task_obj.times),'frames ):'
	dummy_var = task_obj.printTaskDef(1)
	print 'New path (', sum(dummy_task.times),'frames ):'
	new_path_info = dummy_task.printTaskDef(sum(dummy_task.times)/float(sum(task_obj.times)))	#prints the new path information in a good way
	return


def plotFeaturesTogether(data_obj,col,starts,tasknums):
	import matplotlib.pyplot as plt 

	colors = 'kbgrmy'
	for i,t in enumerate(tasknums):
		a,b = starts[t],starts[t+1]
		print a,b 
		print colors[i]
		plt.plot(np.arange(b-a),data_obj.feat_array[a:b,col],'-',color=colors[i],label='task'+str(t))
	plt.legend()


