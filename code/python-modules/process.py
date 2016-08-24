import numpy as np 
import assign
import clusters
import kinectData #maybe unnecessary
import utils
import rayleigh
import task_tools

np.set_printoptions(threshold=np.nan,precision=3,suppress=True)

class Process(object): 
	def __init__(self,init_task): 
		#initialize a process with a task

		#dictionary of known tasks ordered by when they are discovered
		self.known_tasks = {}			
		self.known_tasks[0] = init_task
		self.known_task_count = 1

		#dictionary of mixedRayleigh objects associated with each task 
		self.mixed = {}
		self.mixed[0] = rayleigh.MixedRayleigh(init_task,position=0)

		#keeps track of the current number of frames since the last task completion
		self.curr_frame_count = 0

		#percent complete information for each known task (indexed by taskid)
		self.task_pct_complete = {}		#holds the percent completes for each 
		self.task_pct_complete[0] = 0.0

		#probability information that a particular task is being performed online
		self.task_online_probability = {}
		self.task_online_costs = {}
		self.min_frames_for_probability = 30
		self.online_dtw_constraint = 0.2

		#dynamic time warping cost information for determining which (if any) task to update
		self.task_final_cost = {}

		#historical order of tasks - can be used to adjust probabilities of tasks (ie more recent tasks are more likely)
		self.task_history = [0]	

		#kNN number can be updated according to how many tasks have been completed if wanted
		self.kNN_number = 20	

		#threshold for determining a task to be in the "complete" phase (ie, robot should continue to end)
		self.complete_threshold = 80.0

		#threshold for determining that a new task is unique (determined through experimentation among many task comparisons)
		self.final_dtw_constraint = 0.05		#constraint used on dtw that was used to get the unique task threshold
		self.unique_task_threshold = 3.353

	def onlineUpdate(self,curr_data,data_object,complete=False):

		task_check_order = np.arange(self.known_task_count)		#can maybe do something smart here
		print 'Task check order: ', task_check_order
		#print task_check_order, self.known_tasks
		for t_id in task_check_order: 
			#use the new piece of data to get percent complete for each task model, also update the mixedRayleigh associated with each task 
			tid_task = self.known_tasks[t_id]
			pct_complete,new_mixed = tid_task.getCurrentLabel(curr_data,data_object,self.curr_frame_count,mixed=self.mixed[t_id],kNN_number=self.kNN_number,complete_threshold=self.complete_threshold)
			
			#determine how likely each task is to be the current task by computing dynamic time warping cost between the curr_labels and the expected path at this point
			if self.curr_frame_count > self.min_frames_for_probability: 
				self.task_online_costs[t_id] = task_tools.getTaskMetric(tid_task.path,tid_task.times,tid_task.curr_labels,tid_task.curr_mixed_position,tid_task.frames_since_state_change,constraint=self.online_dtw_constraint)

			self.task_pct_complete[t_id] = pct_complete 
			self.mixed[t_id] = new_mixed
			#print 'pct_complete: ', self.task_pct_complete[t_id]
		
		if self.curr_frame_count > self.min_frames_for_probability and self.known_task_count>1: 
			total_costs = np.sum(self.task_online_costs.values())
			pct_complete = 0
			if total_costs > 0: 
				temp = {}
				for t_id in task_check_order: 
					temp[t_id] = 1-self.task_online_costs[t_id]/total_costs	 
				total_costs = np.sum(temp.values())
				for t_id in task_check_order:
					self.task_online_probability[t_id] = temp[t_id]/total_costs
				print 'task online probabilities: ', self.task_online_probability, self.task_online_costs, total_costs
				for t_id in task_check_order: 
					pct_complete += self.task_online_probability[t_id]*self.task_pct_complete[t_id]
			else: 
				pct_complete = self.task_pct_complete[self.task_history[-1]]
		else: 
			pct_complete = self.task_pct_complete[self.task_history[-1]]

		self.curr_frame_count += 1		#increment the frame count 
		curr_pct_complete = pct_complete 
		return curr_pct_complete

	def updateKnownTasks(self,data_object): 
		#compare latest features to the example task inds features to get median cost from Dynamic Time Warping
		#for each task 
		task_check_order = np.arange(self.known_task_count)
		new_inds = np.arange(data_object.num_vectors-self.curr_frame_count,data_object.num_vectors)
		for t_id in task_check_order: 
			example_inds = self.known_tasks[t_id].task_example_inds
			self.task_final_cost[t_id] = self.compareTasks(data_object,example_inds,new_inds,self.known_tasks[t_id].feature_inds)
		min_cost = np.min(self.task_final_cost.values())
		print 'Min Cost: ', min_cost

		#if min_cost is less than the cost threshold, update the corresponding task, else create a new task 
		if min_cost < self.unique_task_threshold: 
			for t_id in task_check_order: 
				if self.task_final_cost[t_id] == min_cost: 
					print 'Task', t_id, 'updated.'
					self.known_tasks[t_id].update(data_object,[new_inds[0],data_object.num_vectors])
					self.task_history.append(t_id)
					break
		else: 
			print 'new task added.'
			self.known_task_count += 1
			self.known_tasks[self.known_task_count-1] = Task(data_object,[new_inds[0],data_object.num_vectors])
			self.mixed[self.known_task_count-1] = rayleigh.MixedRayleigh(self.known_tasks[self.known_task_count-1])
			task_check_order = np.arange(self.known_task_count)
			self.task_history.append(self.known_task_count-1)
		#reset key task information 
		
		for t_id in task_check_order: 
			self.known_tasks[t_id].curr_labels = []		#the rest of the onlineUpdate specific class variables are updated based on len(curr_labels) == 0

		self.curr_frame_count = 0		#reset number of frames in the new task
		return
			

	def compareTasks(self,data_object,example_inds,new_inds,feature_inds): 
		#collect example data from the data object
		example_data = data_object.all_features[example_inds,:]
		example_data = example_data[:,feature_inds]
		#collect the new data from the data object
		new_data = data_object.all_features[new_inds,:]
		new_data = new_data[:,feature_inds]
		#setup the costs
		costs = ['']*len(feature_inds)
		#perform dynamic time warping on each feature and take the median of the cost associated with each feature
		for i in np.arange(len(feature_inds)): 
			path,costs[i] = task_tools.basisTimeWarp(example_data[:,i],new_data[:,i],constraint=self.final_dtw_constraint)
		return np.median(costs)



class Task(object): 
	def __init__(self, data_object, curr_extrema=[0,10], k=3, basis_dim=2):
		'''
		Purpose: 

		Inputs: 

		Outputs: 

		'''
		
		def defineInitBasisLabels(data_object, curr_extrema, basis_dim, k): 
			curr_input = data_object.feat_array[curr_extrema[0]:curr_extrema[1],:]
			labels, centers, U = clusters.spectralClustering(curr_input, similarity_method=data_object.similarity_method, k_nn=6, basis_dim=basis_dim, num_clusters=k)
			labels = utils.orderStates(labels)
			labels = [int(x) for x in labels]
			labels = list(labels)
			for ind,col in enumerate(U.T): 
				U[:,ind] = assign.stretchBasisCol(col)
			return labels, U

		#onlineUpdate related
		self.feature_inds = data_object.feature_inds					#features of data_object.all_features that are most relevant to this task
		self.data_inds = np.arange(curr_extrema[0],curr_extrema[1])		#tasks are only initialized if it has been determined that a new task model is necessary
		self.task_example_inds = np.arange(curr_extrema[0],curr_extrema[1])	#to be used for comparison to future tasks
		self.curr_labels = []
		
							

		#task related
		self.path = [0,1,2,1,0]				#list of states that the task follows
		self.times = [10,40,100,10,20]		#expected number of frames for the correspondingly indexed path state
		self.states = {}					#dictionary {<state_number>: State instance}
		self.history = []					# all previous data points assigned to this task type 
		self.Taddstate = 0.18				#may want to define this according to how many states exist
		self.basis_dim = basis_dim					#embedding dimension
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
		'''
		Purpose: 

		Inputs: 

		Outputs: 

		'''
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

		self.data_inds = np.hstack((self.data_inds,np.arange(curr_extrema[0],curr_extrema[1])))

		u = self.printTaskDef()
		return

	def getCurrentLabel(self,new_data,data_object,curr_frame_count,mixed,kNN_number=20,complete_threshold=80.0): 
		'''
		Purpose: 
		Takes a new row of data as input and uses kNN, mixedRayleigh proportions, and recent labels to append a new state to the curr_labels of the online task. Also, updates the mixedRaleigh position. 

		Inputs: 
		new_data - 
		data_object - 
		curr_frame_count - 
		mixed - 
		kNN_number - 
		complete_threshold - 

		Outputs: 
		percent_complete - 

		'''
		def updateCounts(count_info,proportions): 
			#print 'PREupdate: ', count_info[0],count_info[1], '* ', proportions
			for ind,state in enumerate(count_info[0]): 
				#print 'Trouble info:', count_info[1][ind], proportions[state]
				count_info[1][ind] = float(count_info[1][ind]) * float(proportions[state])
			#print 'POSTupdate: ', count_info[0],count_info[1]
			return count_info
		def guessFromPast(curr_labels,past_length=3):
			'''
			gets majority vote from past maximum 'past_length' number of labels (or all existing labels) in curr_labels 
			'''
			numlabels = len(curr_labels)
			if numlabels>0:
				if numlabels > past_length:
					consider = curr_labels[-past_length:]
				else: 
					consider = curr_labels 
				#print 'Guess from past function consider: ', consider
				best,aux = utils.majorityVote(consider)
				return best
			else: 
				return -1
		def updatePosition(mixed,path,base_state,new_state,count_new,curr_position,position_threshold = 4): 
			'''
			pseudocode: 
			if base_state == new_state, return count_new = 0, mixed unaltered
			else count_new++, if count_new > position_threshold, increment position and recreate mixed, count_new = 0, base_state = new_state, else return 
			return count_new 
			'''
			#print 'Current rayleigh position: ', curr_position
			if base_state == new_state: 
				count_new = 0 
			else: 
				count_new += 1
				if count_new > position_threshold: 
					#print 'curr/next = ', curr_position, '/',np.argmax(np.array(path[curr_position:])==new_state), 'new: ', new_state
					curr_position += np.argmax(np.array(path[curr_position:])==new_state)
					#print curr_position, 'llllll'
					if curr_position == 0: 
						curr_position = len(path)-1		#if there is not corresponding position, then default to the last position
					mixed.updateSelf(curr_position)		#update the mixedRayleigh
					base_state = new_state
			#print 'New rayleigh position: ', curr_position, 'base state new: ', base_state
			return base_state, count_new, mixed, curr_position
		def taskPercentRemaining(path,times,curr_position,differential): 
			curr_state_time_remaining = max(times[curr_position]-differential, 0)
			#print 'task.times[curr_position]-differential: ', times[curr_position], '-', differential,'=', times[curr_position]-differential
			#print 'Current state time remaining: ', curr_state_time_remaining
			if curr_position == len(path)-1: 
				future_states_times = 0
			else: 
				future_states_times = np.sum(times[(curr_position+1):])
			#print 'Expected future state time remaining: ', future_states_times
			total_task_time = np.sum(times)
			percent_remaining = 100*(curr_state_time_remaining+future_states_times)/float(total_task_time)
			percent_complete = 100 - percent_remaining
			return percent_complete

		#define base state
		if len(self.curr_labels) == 0:
			self.base_state = self.path[0]
			self.count_new = 0 
			self.curr_mixed_position = 0
			self.last_state_change_frame = 0
			self.percent_complete = 0.0
			self.frames_since_state_change = 0
			mixed.updateSelf(self.curr_mixed_position)

		#get labeled data with correct rows from the data object and correct task-specific features
		labeled_data = data_object.all_features[self.data_inds,:]	
		labeled_data = labeled_data[:,self.feature_inds]

		#pick out the newest data
		curr_data = new_data[self.feature_inds]

		#get initial kNN count
		[knn_label,count_info] = utils.kNN(curr_data,labeled_data,self.labels,k=kNN_number) 
		#print 'Initial Knn: ', knn_label, count_info 
		#print 'Trouble1: ', curr_frame_count-self.last_state_change_frame
		proportions = mixed.proportionate(curr_frame_count-self.last_state_change_frame)
		#print 'MixedRayleigh proportions: ', proportions

		# incorporate proportions and past few labels
		if proportions == -1: 
			knn_label_updated = knn_label 
		else: 
			count_info_updated = updateCounts(count_info,proportions)
			#print 'After proportions considerations: ', count_info_updated	
			expectedfrompast = guessFromPast(self.curr_labels)
			#print 'Guess from past: ', expectedfrompast

			# if labels have been added, used expectedfrompast to weight the most likely candidate
			if expectedfrompast != -1: 
				x = count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)]
				x += 1		#this addition and multiplication gives a bit of a chance to low scoring values
				x *= 1.1
				count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)] = x

			# determine new label based on weighted kNN 
			#print 'After guess from past applied: ', count_info_updated
			knn_label_updated = count_info_updated[0][np.argmax(count_info_updated[1])]
		#print 'New knn label chosen: ', knn_label_updated
		self.curr_labels.append(knn_label_updated)

		# update mixedRayleigh distribution for the new frame/possibly new base state 
		new_base_state,new_count_new,new_mixed,new_curr_mixed_position = updatePosition(mixed,self.path,self.base_state,knn_label_updated,self.count_new,self.curr_mixed_position,position_threshold=4)

		# update global variables

		self.base_state = new_base_state
		self.count_new = new_count_new
		if self.curr_mixed_position != new_curr_mixed_position:
			mixed_position_changed = True
			self.last_state_change_frame = curr_frame_count
			self.curr_mixed_position = new_curr_mixed_position

		# get the percent complete. if 
		self.frames_since_state_change = curr_frame_count-self.last_state_change_frame
		#print 'Frames since state change: ', frames_since_state_change
		self.percent_complete = taskPercentRemaining(self.path,self.times,self.curr_mixed_position,self.frames_since_state_change)

		return self.percent_complete, new_mixed



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
		if len(G_acc) == 1: 
			return np.array([1.])
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

