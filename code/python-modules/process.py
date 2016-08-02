import numpy as np 
import assign
import clusters
import kinectData #maybe unnecessary
import utils

np.set_printoptions(threshold=np.nan,precision=3,suppress=True)

class Process(object): 
	def __init__(self): 
		# kinectData objects for the parties 
		# midpoint information
		# 
		return

	def matchTask(self,curr_data,complete=False):
		''' Compares what is in curr_data to tasks known by the process and returns probabilities that curr_data belongs to each known class. complete=False refers to whether the task has been completed and a final decision needs to be made that the task is novel or it was known. If complete=True, the task is added to process history. '''
		return 


class Task(object): 
	def __init__(self, data_object, curr_extrema=[0,10], k=3):
		
		def defineInitBasisLabels(data_object, curr_extrema, basis_dim, k): 
			curr_input = data_object.feat_array[curr_extrema[0]:curr_extrema[1],:]
			labels, centers, U = clusters.spectralClustering(curr_input, similarity_method=data_object.similarity_method, k_nn=6, basis_dim=basis_dim, num_clusters=k)
			labels = utils.orderStates(labels)
			labels = [int(x) for x in labels]
			labels = list(labels)
			for ind,col in enumerate(U.T): 
				U[:,ind] = assign.stretchBasisCol(col)
			return labels, U

		self.path = [0,1,2,1,0]				#list of states that the task follows
		self.times = [10,40,100,10,20]		#expected number of frames for the correspondingly indexed path state
		self.states = {}					#dictionary {<state_number>: State instance}
		self.history = []					# all previous data points assigned to this task type 
		self.Taddstate = 0.18				#may want to define this according to how many states exist
		self.basis_dim = 2					#embedding dimension
		self.k = k							#number of clusters
		self.Tdistwidth = 1.5				#start with requiring new points to be within 1.5 standard deviations of each class definition to be considered inside the class

		labels, U = defineInitBasisLabels(data_object,curr_extrema,self.basis_dim,self.k)
		self.subspace = utils.Subspace(U)
		G_acc = self.defineAcceptedStates(U, labels, self.Tdistwidth)
		self.setupStates(G_acc)
		self.split_points = self.findSplitPoints(G_acc)
		#print self.split_points
		self.labels = self.redistributePoints(U,self.split_points)
		self.refineStates(U,self.labels)
		self.history.append(U[:,0])
		[path,times]=self.definePath(self.labels)
		self.path = path
		self.times = times
		self.split_points = self.findSplitPoints(self.getStateInfo()[0])
		#print self.split_points

	def update(self, data_object, curr_extrema=[0,10]): 
		''' For use when a task is complete and the task models need to be updated. The task which Process.matchTask() has determined this task to be is the one being updated with the new data for the entire task process that has just occured. '''
		def embed(data_object,curr_extrema=[0, 10]): 
			curr_input = data_object.feat_array[curr_extrema[0]:curr_extrema[1],:]
			U = clusters.getLaplacianBasis(curr_input,similarity_method=data_object.similarity_method,k_nn=6)
			newU = self.subspace.projectOnMe(U)
			for ind,col in enumerate(newU.T): 
				newU[:,ind] = assign.stretchBasisCol(col)
			return newU

		def addToHistory(U): 
			self.history.append(U[:,0])
			ufull = np.array([])
			for u in self.history: 
				ufull = np.hstack((ufull,u))
			ufull = ufull.reshape(len(ufull),1)
			Ufull = np.hstack((ufull,np.ones_like(ufull)))
			return Ufull

		U = embed(data_object,curr_extrema)
		Ufull = addToHistory(U)

		# throw everything onto the basis and use the previous split points to say where everything is now
		labels = self.redistributePoints(Ufull,self.split_points) 
		self.Tdistwidth -=  0.2*len(self.history)**(-1)
		if self.Tdistwidth < 1.: 
			self.Tdistwidth = 1.
		#print 'distwidth (threshold): ', self.Tdistwidth
		# now, with the new labels and the basis values, define new states, then do splitStates() to determine if a new one should be added 
		G_acc = self.defineAcceptedStates(Ufull, labels, self.Tdistwidth)
		# setup the new states for this Task object
		self.setupStates(G_acc)
		# find the split points of these accepted states, and redistribute accordingly
		self.split_points = self.findSplitPoints(G_acc)
		#print self.split_points
		self.labels = self.redistributePoints(Ufull,self.split_points)

		#now that the points are redistributed, I now can update my state definitions 
		self.refineStates(Ufull, self.labels)
		self.split_points = self.findSplitPoints(self.getStateInfo()[0])
		# the n handovers i have now witnessed should follow the same path, so the labels I now have are for n processes, I need to split these processes up and check that they are the same. 

		start_ind = 0
		task_count = len(self.history)
		path_history = [0]*task_count
		times_history = [0]*task_count

		#determine the path and times using this set of states for each historical task of this type 
		for i,u in enumerate(self.history): 
			u = u.reshape(len(u),1)
			l = self.labels[start_ind:(start_ind+len(u))]
			start_ind += len(u)
			[path_history[i],times_history[i]] = self.definePath(l)

		#determine the most often len(path_history[i]) and use that as the new path for the class, update the times with the relevant times_history[i]'s
		path_lengths = [len(p) for p in path_history]
		#print 'pathlens: ', path_lengths
		uValues = np.unique(path_lengths).tolist()		#unique path lengths of all the tasks in path history
		#print 'unique path length values   : ', uValues
		uCounts = [len(np.nonzero(np.array(path_lengths) == uv)[0]) for uv in uValues]	#the counts associated with each of the unique path lengths
		#print 'cnts of unique path lengths : ', uCounts
		uCountsArr = np.array(uCounts)			#array version of uCounts
		counts_index = np.argmax(uCountsArr)	#index of the largest value of the path lengths 
		key_inds = np.where(np.array(path_lengths) == uValues[counts_index])[0]		#indices of path_history/times_history associated with the most consistent path length
		#print 'inds of best path lengths: ', key_inds
		path_choice = path_history[key_inds[0]]
		#for each times_path at the key_inds in times_history, add them up
		times_total = [0]*len(path_choice)	#initialize to correct size
		for i,k in enumerate(key_inds): 
			for j in np.arange(len(times_total)):
				times_total[j] += times_history[k][j]

		times_avg = [t/(1.0*np.sum(np.array(path_lengths)==uValues[counts_index])) for t in times_total]	

		self.times = [int(x) for x in list(times_avg)]
		self.path = path_choice 

		u = self.printTaskDef()
		return

	def printTaskDef(self,str_scaler=0.75): 
		totalTime = np.sum(self.times)
		str_scaler = str_scaler		# string output to console is str_scaler*100 chars long
		props = [x/(0.01*totalTime) for x in self.times]
		output = ''
		for i,p in enumerate(props): 
			half = int((0.5*p-1)*str_scaler)
			output = output + '['+half*'-'+str(self.path[i])+half*'-'+']'
		print output
		return [output, props]

	def getStateInfo(self): 
		G = np.zeros((self.num_states,2)) 		#[mean, std]
		print 'stateshape: ', G.shape, 'numstates: ', self.num_states
		c = np.zeros(self.num_states)
		tpoints = 0
		for s in self.states: 
			G[s,0] = self.states[s].mean
			G[s,1] = self.states[s].std
			c[s] = self.states[s].npoints
			tpoints += c[s]
		c = c/(1.*tpoints)
		return [G,c]


	'''
	---------------------------------
	------auxilliary functions-------
	---------------------------------
	'''
	def defineAcceptedStates(self, U, labels, Tdistwidth): 
		# get class dists initially
		G = assign.getClassDists(U[:,0],labels)
		# test for a missing state
		G_new, counts = assign.splitStates(G,U[:,0],Tdistwidth)
		# which are accepted based on the Tdistwidth threshold
		totalCounts = np.sum(counts)
		props = [c/(1.*totalCounts) for c in counts]
		Taddstate_mult = 0.75
		Taddstate = np.sort(props)[-3]*Taddstate_mult
		accepted = np.array([x for x in np.arange(len(counts)) if props[x]>Taddstate])
		G_acc = G_new[accepted,:]
		G_acc = G_acc[G_acc[:,0].argsort()]
		self.num_states = len(accepted)
		self.Taddstate = Taddstate
		#print 'addstate  (threshold): ', self.Taddstate
		#print 'proportions: ', props
		return G_acc

	def setupStates(self, G_acc):
		self.states = {}
		self.num_states = len(G_acc)
		for i,n in enumerate(G_acc): 
			key = i
			value = State(n[0],n[1])
			self.states[key]=value
		return

	def findSplitPoints(self, G_acc): 
		split_points = np.zeros(len(G_acc)-1)
		for a in np.arange(len(G_acc)-1): 
			root = utils.gaussiansMeet(G_acc[a,0],G_acc[a,1],G_acc[a+1,0],G_acc[a+1,1])
			root = root[np.logical_and(root>G_acc[a,0],root<G_acc[a+1,0])]	#assure that only the relevant split point is chosen by having the value be between the two means of the guassians compared
			split_points[a] = root
		split_points = np.insert(split_points,[0,len(split_points)],[-1,1])		#tack on -1 and 1 to the ends of the segment definition
		return split_points

	def redistributePoints(self, U, split_points): 
		labels = np.ones(len(U))*-1
		for i,s in enumerate(split_points): 
			if i == len(split_points)-1: 
				break
			for j,u in enumerate(U[:,0]): 
				if labels[j] >= 0: 
					continue
				else: 
					if np.logical_and(u>=s,u<=split_points[i+1]): 
						labels[j] = i
						self.states[i].npoints += 1
		return labels

	def refineStates(self,U,labels): 
		G = assign.getClassDists(U[:,0],labels)
		for i,g in enumerate(G): 
			self.states[i].mean = g[0]
			self.states[i].std = g[1]
		return

	def definePath(self,labels): 
		labels1 = list(labels)
		labels2 = list(np.hstack((labels,labels[-1])))
		path = [x for ind,x in enumerate(labels1) if x != labels2[ind+1]]
		path.append(labels1[-1])
		path = [int(x) for x in path]
		#print '*************************************'
		#print 'PATH: ',path
		times = [0]*len(path)
		#print 'times; ', times
		#print '*************************************'
		for i,p in enumerate(path):
			so_far = int(np.sum(times[0:i]))
			if i == 0: 
				times[i] = np.argmax(np.array(labels)!=p)
			elif i < (len(times)-1): 
				times[i] = np.argmax(np.array(labels[so_far:])!=p)
			else: 
				times[i] = len(labels[so_far:])
		return [path, times]

class State(object): 
	def __init__(self,mean,std): 
		self.mean = mean
		self.std = std
		self.npoints = 0

def exampleSetup(): 
	import test
	data4,data5,handover_starts,l0,U0,l1,U1e,U1,l2,U2,U2e,ainds,alabs = test.main()

	def loader(handover_starts,data_object,n): 
		try: 
			starts = [handover_starts[n],handover_starts[n+1]]
		except IndexError: 
			starts = [handover_starts[n],data_object.num_vectors-1]
		return starts

	def runTasks(data_obj,task_obj,n,max_ind=10): 
		inds = np.random.randint(max_ind,size=n)
		for i in inds: 
			task_obj.update(data_obj,loader(handover_starts,data_obj,i))

	#initializing a task for the data4 object then showing the times for each path member
	task = Task(data4,loader(handover_starts,data4,0))
	print task.times
	out = task.printTaskDef()

	#updating a task for the data4 object then showing the times for each path member
	task.update(data4,loader(handover_starts,data4,1))
	print task.times
	out = task.printTaskDef()

	#or can do: 
	task = Task(data4,loader(handover_starts,data4,0))
	n = 20 #run 20 random updates from the handover set
	runTasks(data4,task,n,max_ind=11)





	



if __name__== '__main__': setup()

