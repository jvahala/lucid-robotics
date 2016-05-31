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
	for dist_num,dist in enumerate(gaussians): 
		for ind,elt in enumerate(vector): 
			diff = elt-dist[0]
			if (np.abs(diff)<dist[1]*thresh_mult): 
				new_labels[ind] = dist_num
			elif np.sign(diff)==1: 		#if elt is right of the current distribution mean 
				new_labels[ind] -= 1	# [ (-1) 0 (-2) 1 (-3) 2 (-4) 3 (-5) ...] where (-k) is a class label denoting the point to be inbetween true classes
	#rename states to be >= 0 (-1->0, (-1-k)->numclasses+k-1, minind (negative)->numclasses-1)
	new_labels = [0 if x==-1 else num_classes-1 if x==-1*(num_classes+1) else (np.sign(x)*x+gaussians.shape[0]-2) if x<0 else x for x in new_labels]
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
	plt.show()

def plotClassPoints(basis,labels):
	import matplotlib.pyplot as plt 
	num_classes = getNumClasses(labels)
	'''plt.figure(1)
	for state in np.arange(num_classes):
		plt.plot(basis[labels==state,0]/np.amax(np.abs(basis[:,0])),basis[labels==state,1]/np.amax(np.abs(basis[:,1])),marker='x',linestyle='None')
	plt.xlabel('basis 1')
	plt.ylabel('basis 2')
	plt.legend(['class '+str(state) for state in np.arange(num_classes)])
	plt.axis([-1.1,1.1,0.5,1.5])'''

	plt.figure(2) 
	color_options = ['r','g','b','y','m','k']
	normalizer = np.amax(np.abs(basis[:,0]))
	time_scalar = 100/(1.0*len(labels))
	for i,x in enumerate(labels): 
		curr_col = color_options[x]
		plt.plot(i*time_scalar,basis[i,0]/normalizer,marker='x',color=curr_col)
	'''counts = [np.sum(labels==x) for x in np.arange(num_classes)]
	start = 0
	for state in np.arange(num_classes):
		end = start+counts[state]
		x=np.arange(start,start+counts[state])
		plt.plot(x, basis[labels==state,0]/np.amax(np.abs(basis[:,0])),marker='x',linestyle='None')
		start = end'''
	plt.xlabel('index')
	plt.ylabel('basis 1 value')
	plt.legend(['class '+str(state) for state in np.arange(num_classes)])
	plt.axis([0,100,-1.1,1.1])


