'''
module: assign.py 
use: contains functions associated with assigning new states and updating class descriptions
'''

import numpy as np

def makeGuassian(vector):
	'''
	Purpose: 
	turns one dimensional data into a guassian distribution with mean mu and variance var = std^2 

	Inputs: 
	vector - 1 dimensional array of values

	Outputs: 
	gauss - tuple (mu,sigma) - (mean of the input vector, std of the input vector)
	'''
	mu = np.mean(vector)
	sigma = np.std(vector)
	gauss = (mu,sigma)
	return gauss

def getClassDists(vector,labels):
	'''
	Purpose: 
	turns vector-label pairs into however many sets of gaussian distributions 

	Inputs: 
	vector - 1 dimensional array of values
	labels - list of labels assigned to vector values (same size as vector)

	Outputs: 
	gaussians - num_classes by 2 array of [mu,sigma] - (mean of the input vector, std of the input vector)
	'''
	num_classes = getNumClasses(labels)
	gaussians = np.zeros((num_classes,2))
	for i in np.arange(num_classes):
		curr_class = vector[labels==i]
		gaussians[i,:] = np.array(makeGuassian(curr_class))
	return gaussians

def getNumClasses(labels):
	'''
	Purpose: 
	gets the number of classes (unique numbers in labels): num_classes = highest class number + 1

	Inputs: 
	labels - list of classes assigned to each frame

	Outputs: 
	integer number of classes in labels
	
	'''
	return np.amax(labels)+1; 

def splitStates(gaussians, vector, thresh_mult=1): 
	'''
	Purpose: 
	splits points in vector into classes of points. If a point is within one threshold amount (one option for threshold is the std of the class) of the class mean , it is accepted as in the class. This function returns the number of points in each split class as well as the new means and variances of the classes. 

	Inputs: 
	gaussians - array of gauss (mu,sigma) tuples representing the distributions defined by the current classes
	vector - one dimensional array of points 
	thresh_mult - multiplier to move the treshold closer (lower than 1) or further (greater than 1) than the one standard deviation point

	Outputs: 
	new_gaussians - num_classes by 2 array of [mu,sigma,label] pairs
	counts - number of elements in each new potential class 
	'''
	new_labels = -1*np.ones(len(vector))
	num_classes = gaussians.shape[0]
	gaussians = gaussians[gaussians[:,0].argsort()]		#sort the gaussians by mean [most negative to least negative mean]
	print new_labels
	for dist_num,dist in enumerate(gaussians): 
		for ind,elt in enumerate(vector): 
			diff = elt-dist[0]
			if (np.abs(diff)<dist[1]*thresh_mult): 
				new_labels[ind] = dist_num
			elif np.sign(diff)==1: 		#if elt is right of the current distribution mean 
				new_labels[ind] -= 1	# [ (-1) 0 (-2) 1 (-3) 2 (-4) 3 (-5) ...] where (-k) is a class label denoting the point to be inbetween true classes
	#rename states to be >= 0 (-1->0, (-1-k)->numclasses+k-1, minind (negative)->numclasses-1)
	new_labels = [int(x) for x in new_labels]
	print new_labels
	#-2 -> max_class+1, -3->max_class+2, -4->max_class+3
	new_labels = [int(0) if x==-1 else int(num_classes-1) if x==-1*(num_classes+1) else int(num_classes+(-2-x)) if x<-1 else int(x) for x in new_labels]
	print num_classes,new_labels
	num_classes = getNumClasses(new_labels)
	new_gaussians = getClassDists(vector,new_labels)
	counts = [np.sum(new_labels==x) for x in np.arange(num_classes)]
	'''print 'new labels: ', new_labels 
	print 'old gauss: \n', gaussians
	print 'new gauss: \n', new_gaussians
	print 'counts: ', counts 
	print 'percents: '
	for count in counts: 
		print count/(1.0*sum(counts))
	plotClassDists(new_gaussians,counts)'''

	return new_gaussians, counts

def stretchBasisCol(basis_col): 
	maxpos = np.amax(basis_col)
	maxneg = np.abs(np.amin(basis_col))
	for i,u in enumerate(basis_col):
		if u >= 0: 
			basis_col[i] = u/maxpos
		else: 
			basis_col[i] = u/maxneg
	return basis_col 

def plotClassDists(gaussians,multiplier):
	'''
	Purpose: 
	plots the guassian distributions defined by the n by 2 gaussian array of (mu, sigma) variables with a multiplier to the distribution heights if wanted

	Inputs: 
	gaussians - n by 2 array of gauss (mu,sigma) elements representing the distributions defined by the current classes
	multiplier - length n object of multiplier for each index of gaussians 

	Outputs: 
	plot of the gaussian distributions
	'''
	import matplotlib.mlab as mlab
	import matplotlib.pyplot as plt 
	x = np.linspace(-2,2,100)
	for dist_num,dist in enumerate(gaussians): 
		plt.plot(x,multiplier[dist_num]/(1.0*sum(multiplier))*mlab.normpdf(x,dist[0],dist[1]))

def plotClassPoints(basis,labels):
	import matplotlib.pyplot as plt 
	num_classes = getNumClasses(labels)
	'''plt.figure(1)t
	for state in np.arange(num_classes):
		plt.plot(basis[labels==state,0]/np.amax(np.abs(basis[:,0])),basis[labels==state,1]/np.amax(np.abs(basis[:,1])),marker='x',linestyle='None')
	plt.xlabel('basis 1')
	plt.ylabel('basis 2')
	plt.legend(['class '+str(state) for state in np.arange(num_classes)])
	plt.axis([-1.1,1.1,0.5,1.5])'''

	color_options = ['r','g','b','y','m','k']
	time_scalar = 100/(1.0*len(labels))
	plt.plot(np.arange(len(labels))*time_scalar, basis-0.05,linestyle='-',marker='None',color='0.75')
	for i,x in enumerate(labels): 
		curr_col = color_options[x]
		plt.plot(i*time_scalar,basis[i],marker='x',color=curr_col)
	'''counts = [np.sum(labels==x) for x in np.arange(num_classes)]
	start = 0
	for state in np.arange(num_classes):
		end = start+counts[state]
		x=np.arange(start,start+counts[state])
		plt.plot(x, basis[labels==state,0]/np.amax(np.abs(basis[:,0])),marker='x',linestyle='None')
		start = end'''
	plt.xlabel('scaled index')
	plt.ylabel('basis 1 value')
	plt.axis([0,100,-1.1,1.1])

def updateClasses(prev_vector,prev_labels,vector,labels,thresh_mult=1,thresh_prop=0.2,min_classes=3): 
	'''
	Purpose: 
	updates currently defined classes to refine state descriptions
	Class addition: if currently in 3 classes, will find the distribution of the three classes within (thresh_mult*std_of_guassian) then define new classes inbetween the old classes. If those new class points represent some strong proportion (thresh_prop) of the total number of classes, then the new classes will be added as unique to the min_classes classes that have the strongest representation. 
	Class distribution update: Points with the strongest probabilities based on hypothesis testing between classes (normal distributions scaled by the percentage of points within that class) are redistributed to each class to define new class distributions. These distributions are then returned with the new concatenated vector and labels. 

	Inputs: 
	prev_vector - 
	prev_labels - 
	vector - 
	labels - PROBABLY NOT NECESSARY
	thresh_mult - see splitStates() for description (describes what consitutes a unique state)
	thresh_prop - in (0,1], percentage of all points that are within a unique state. If proportion is greater than this threshold, then it is accepted as a unique state
	min_classes - in {1,2,...,(2*num_states-1)} where num_states is the number of unique states at the start of the function call. This is the minimum number of classes to output even if the classes do not meet the required thresh_prop

	Outputs: 
	new_vector - concatenated vector of basis points between [0,1] representing all observed handovers to this point
	new_labels - labels associated with new_vector
	new_gaussians - updated class gaussian descriptions as ndarray n by 2 where n = number of classes 
	new_counts - list of number of elements counts associated with each of the n new_guassians
	'''
	return ''
