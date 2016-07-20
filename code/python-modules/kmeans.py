'''
module: kmeans.py
use: contains functions associated with kmeans classification 
'''
import numpy as np

def kplusplus(X,k): 
    '''
	Purpose: 
	performs kmeans++ algorithm on data X (dim by n) into k centers

	Inputs: 
	X - (dim by n) ndarray of data, so input data to function as basis_vectors.T 
	k - number of clusters to sort X into 

	Outputs: 
	L - labels (1 by n)for each of the n data points of X
	C - cluster centers array (dim by k) representing each of the k cluster centers 

	'''
    L = np.zeros((1,X.shape[1]),dtype = np.int_)  #initialize the various assignments of the cluster ids to the data
    L1 = -1*np.ones((1,X.shape[1]),dtype = np.int_)
    C = np.zeros((X.shape[0],k))  #initialize k centers 
    iterations = 0

    while len(np.unique(L)) != k: #repeat until k init centers chosen
    
        #choose first center L[0] at random from the n columns of X
        C[:,0] = X[:,np.random.choice(X.shape[1])]

        #compute the distances from each point to nearest center
        for i in range(1,k):
            m = 0
            Cl = C[:,0] #initialized starting point
            m=1
            for l in L.reshape(X.shape[1],):
                Cl = np.vstack((Cl,C[:,l])) #stack each center at index l into an array 
                m += 1
            Cl = Cl[1:,:].T #remove the initialized starting point
            D = X - Cl  #compare current center guesses
            D = np.cumsum(np.sqrt(np.diag(D.T.dot(D))))

            
            if D[-1] == 0: #this is rare
                for ix in range(i,k):
                    C[:,ix]=X[:,0]
                return L, C

            C[:,i] = X[:,np.argmax(np.random.rand()<(D/D[-1]))] #initialize the current center index randomly according to probabilies
            val = np.diag(C.T.dot(C)).reshape(C.shape[1],1)
            temp = 2*np.real(C.T.dot(X)) - val #the max of rows of this determine which cluster is closest
            L = np.argmax(temp,axis=0)

        #update cluster centers
        while np.any(L1 != L) and iterations <1000: 
            L1 = L 
            for i in range(0,k): 
                l = (L == i) #get boolean array of spots where the feature is assigned to cluster i
                C[:,i] = np.sum(X[:,l],axis=1)/np.sum(l) #get new average center 
            val = np.diag(C.T.dot(C)).reshape(C.shape[1],1)
            temp = 2*np.real(C.T.dot(X)) - val
            L = np.argmax(temp,axis=0) #recalculate the cluster id's to see if any of changed with the new cluster center identifications
            iterations +=1
            #print iterations
    return L.reshape(L.shape[0],), C

def example():
	#import necessary modules to run example
	from utils import generateData
	import matplotlib.pyplot as plt 

	#generate data
	num_points = 100
	X, y = generateData(num_points,2,'bull')

	#perform kmeans on raw data
	L,C = kplusplus(X.T,2)
	print 'labels: ', L, '\nCenters: \n', C

	#display results
	plt.figure(1)
	plt.subplot(121,aspect='equal')
	plt.plot(X[y==+1,0],X[y==+1,1],linestyle='None',marker='x',color='r',label='Generated +1 labels')
	plt.plot(X[y==-1,0],X[y==-1,1],linestyle='None',marker='x',color='b',label='Generated -1 labels')
	plt.legend(loc=4,fontsize='x-small',numpoints=1)
	plt.title('Raw Data')
	plt.axes([-0.5 ,1.5 ,-0.5 ,1.5])
	
	plt.subplot(122,aspect='equal')
	plt.plot(X[L==0,0],X[L==0,1],linestyle='None',marker='o',color='r',label='Group 1 cluster')
	plt.plot(X[L==1,0],X[L==1,1],linestyle='None',marker='o',color='b',label='Group 2 cluster')
	plt.legend(loc=4,fontsize='x-small',numpoints=1)
	plt.title('Clustered Data')

	plt.show()  
