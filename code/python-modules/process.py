import numpy as np 
import assign
import clusters
import kinectData #maybe unnecessary
import utils
import rayleigh
import task_tools
from copy import deepcopy

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
		self.curr_pct_complete_estimate = 0.0 	#probabilistic estimate of percent complete metric

		#probability information that a particular task is being performed online
		self.task_online_probability = {}
		self.task_online_costs = {}
		self.min_frames_for_probability = 20
		self.online_dtw_constraint = 0.01
		self.cumulative_online_errors = {}
		self.error_per_frame = {}
		self.error_metric = {}
		self.prev_error_per_frame = {}
		self.confusion_pct_addition = 0.5
		self.error_window = 30.0 

		#dynamic time warping cost information for determining which (if any) task to update
		self.task_final_cost = {}

		#historical order of tasks - can be used to adjust probabilities of tasks (ie more recent tasks are more likely)
		self.task_history = [0]	
		self.max_task_history_to_consider = 30		#number of recent task examples to consider as potential candidates for the new task (set to -1 to use full task history)
		self.task_check_order = self._determineTaskOrder()		#prioritizes most recently seen tasks - doesnt care for tasks seen long long ago

		#kNN number can be updated according to how many tasks have been completed if wanted, this is specific to each task type
		self.init_kNN_number = 20
		self.kNN_number = {0:self.init_kNN_number}	
		self.max_kNN_number = 100
		self.kNN_increment = 15

		#threshold for determining a task to be in the "complete" phase (ie, robot should continue to end)
		self.complete_threshold = 80.0

		#threshold for determining that a new task is unique (determined through experimentation among many task comparisons)
		self.final_dtw_constraint = 0.05		#constraint used on dtw that was used to get the unique task threshold
		self.unique_task_threshold = 1.79 			#1.79 from personal_data collection, ~3.353 from giver vs. receiver collection

	def onlineUpdate(self,curr_data,data_object,complete=False,softmax_error=True,self_correct=True):

		def getErrorMetric(epf,p=None,i=None,d=None,soft=None,last_epf=None,all_err=None):
			'''
			epf = error_per_frame = cumulative_error/current_frame (potentially a windowed epf to allow for quicker changes late in a task)
			p = number of known tasks, also the value of the proportional gain 
			all_err = cumulative error for integral control
			soft = error to softthreshold within 
			last_epf = previous value of epf for derivative control 


			Em = P*epf + D*(epf-last_epf) + I(all_err)
			metric = max(1,Em), or if p,last_epf,and all_err are left as default None then epf is returned
			'''
			P = p 		#proportional gain
			D = d		#derivative gain
			I = i		#integral gain

			Em = 0
			control_terms = 0
			#print 'epf = ', epf
			#Proportional error addition
			if p != None: 
				if soft != None: 
					Em += P*(epf-soft-1)
				else:
					Em += P*(epf-1)
				#print 'prop Em = ', Em
			else: 
				#print 'p = None'
				control_terms += 1

			#Derivative error addition
			if last_epf != None:
				Em += D*(epf-last_epf)
				#print 'deriv Em = ', Em
			else: 
				#print 'last epf = none'
				control_terms += 1

			#Integral error addition
			if all_err != None: 
				Em += I*all_err
				#print 'integral Em = ', Em
			else:
				#print 'all err = none'
				control_terms += 1

			#if no P,I,or D, return the error per frame alone
			if control_terms == 3: 
				Em = epf

			metric = max(1.0,Em)
			return float(metric)
	

		## use the new piece of data to get percent complete for each task model, also update the mixedRayleigh associated with each task
		for t_id in self.task_check_order:  
			tid_task = self.known_tasks[t_id]
			pct_complete,new_mixed = tid_task.getCurrentLabel(curr_data,data_object,self.curr_frame_count,mixed=self.mixed[t_id],kNN_number=self.kNN_number[t_id],complete_threshold=self.complete_threshold)
			
			'''#determine how likely each task is to be the current task by computing dynamic time warping cost between the curr_labels and the expected path at this point - this was too slow
			cost_threshold = 2
			if self.curr_frame_count > self.min_frames_for_probability: 
				self.task_online_costs[t_id] = max(1.0,task_tools.getTaskMetric(tid_task.path,tid_task.times,tid_task.curr_labels,tid_task.curr_mixed_position,tid_task.frames_since_state_change,constraint=self.online_dtw_constraint)-cost_threshold)
				if self.task_online_costs[t_id]>1.0: 
					self.task_online_costs[t_id] = 0.5*self.task_online_costs[t_id]**2
			'''

			#add current online error (or add new element if a new task has just been added)
			expected_pct = 100*self.curr_frame_count/float(np.sum(self.known_tasks[t_id].times))
			c = 0.0625
			if len(self.cumulative_online_errors) < self.known_task_count: 
				self.cumulative_online_errors[t_id] = c*np.abs(expected_pct-pct_complete)**2
				self.prev_error_per_frame[t_id] = 0
				self.error_per_frame[t_id] = self.cumulative_online_errors[t_id]/float(self.curr_frame_count)
			else: 
				new_error = c*np.abs(expected_pct-pct_complete)**2
				self.cumulative_online_errors[t_id] += new_error
				self.prev_error_per_frame[t_id] = deepcopy(self.error_per_frame[t_id])
				base_epf = self.cumulative_online_errors[t_id]/float(self.curr_frame_count)
				error_window = self.error_window
				self.error_per_frame[t_id] = (base_epf*error_window+new_error)/(error_window+1)	#essentially, (curr_epf*error_window + new_error)/(frame_window+1), this allows for more change later in a task 

			self.task_pct_complete[t_id] = pct_complete 
			self.mixed[t_id] = new_mixed
			#print 'pct_complete: ', self.task_pct_complete[t_id]
		
		## if enough frames have passed and a task exists, determine the probability that a particular task is being completed relative to other tasks
		if self.curr_frame_count > self.min_frames_for_probability and self.known_task_count > 0: 
			#total_costs = np.sum(self.task_online_costs.values())
			total_costs = self.known_task_count +1
			pct_complete = 0
			if total_costs > self.known_task_count: 
				self.error_metric = {}
				temp = {}

				#adjust errors using PID to extremify errors further
				for t_id in self.task_check_order: 
					
					P = self.known_task_count
					I = 0.1		#0.02
					D = 50			#100
					max_soft = 10
					#time_at_4 = 30.0		#these two commented lines were used to determine the soft_thresh exponential constant (-0.0305)
					#soft_thresh = max_soft*np.exp((np.log(4/max_soft)/time_at_4)*self.curr_frame_count)
					soft_thresh = max_soft*np.exp(-0.0305*self.curr_frame_count)
					
					last_err = self.prev_error_per_frame[t_id]
					cumulative_error = self.cumulative_online_errors[t_id]

					self.error_metric[t_id] = getErrorMetric(epf=self.error_per_frame[t_id],p=P,i=I,d=D,soft=soft_thresh,last_epf=last_err, all_err=cumulative_error)
					#print self.error_metric[t_id]

					temp[t_id] = 1/self.error_metric[t_id]

				if softmax_error == False: 
					total_costs = np.sum(temp.values())
					for t_id in self.task_check_order: 
						self.task_online_probability[t_id] = temp[t_id]/(1.*total_costs)
				else: 
					metrics = [self.error_metric[x] for x in range(self.known_task_count)]
					softmax_alpha = -1/40.0		#this alpha has bee chosen to that reasonable errors in the 50
					soft_pcts = utils.softmax(np.array(metrics)-np.amin(metrics)/3.0,alpha=softmax_alpha,rescale_=False)
					for t_id in self.task_check_order: 
						self.task_online_probability[t_id] = soft_pcts[t_id]

				if self_correct and np.all(np.array(self.error_metric.values())>200): 
					#print 'in self-correct loop'
					confusion_pct_addition = self.confusion_pct_addition	#amount to add to pct_complete estimate if all errors are bad
					pct_complete = self.curr_pct_complete_estimate + confusion_pct_addition
				else: 
					#print 'no self-correction'
					for t_id in self.task_check_order: 
						pct_complete += self.task_online_probability[t_id]*self.task_pct_complete[t_id]
			else: 
				pct_complete = self.task_pct_complete[self.task_history[-1]]
		else: 
			pct_complete = self.task_pct_complete[self.task_history[-1]]

		self.curr_frame_count += 1		#increment the frame count 
		self.curr_pct_complete_estimate = pct_complete 
		return pct_complete

	def updateKnownTasks(self,data_object,compute_type='median',proper_update=True): 
		#compare latest features to the example task inds features to get median cost from Dynamic Time Warping
		#for each task 

		new_inds = np.arange(data_object.num_vectors-self.curr_frame_count,data_object.num_vectors)
		for t_id in self.task_check_order: 
			example_inds = self.known_tasks[t_id].task_example_inds
			self.task_final_cost[t_id] = self.compareTasks(data_object,example_inds,new_inds,self.known_tasks[t_id].feature_inds,compute_type=compute_type)
		min_cost = np.min(self.task_final_cost.values())
		#print 'Min Costs: ', self.task_final_cost

		#if min_cost is less than the cost threshold, update the corresponding task, else create a new task 
		if min_cost*0.66 < self.unique_task_threshold: 
			for t_id in self.task_check_order: 
				if self.task_final_cost[t_id] == min_cost: 
					if proper_update: 
						self.known_tasks[t_id].update(data_object,[new_inds[0],data_object.num_vectors])
					self.task_history.append(t_id)
					#print 'Task', t_id, 'updated.\n'
					break
		else: 
			self.known_task_count += 1
			self.known_tasks[self.known_task_count-1] = Task(data_object,[new_inds[0],data_object.num_vectors])
			self.mixed[self.known_task_count-1] = rayleigh.MixedRayleigh(self.known_tasks[self.known_task_count-1])
			self.task_history.append(self.known_task_count-1)
			#print 'New Task '+str(self.known_task_count-1)+' added.\n'
		#reset key task information 
		
		for t_id in self.task_check_order: 
			self.known_tasks[t_id].curr_labels = []		#the rest of the onlineUpdate specific class variables are updated based on len(curr_labels) == 0

		self.task_check_order = self._determineTaskOrder()	#update the order in which tasks are checked based on recent history
		#print 'Task Check Order: ', self.task_check_order
		self.curr_frame_count = 0		#reset number of frames in the new task
		self.cumulative_online_errors = {}

		try: 
			if self.kNN_number[self.task_history[-1]] <= self.max_kNN_number: 
				self.kNN_number[self.task_history[-1]] = min(self.max_kNN_number,self.kNN_number[self.task_history[-1]]+self.kNN_increment)	#increment the number of nearest neighbors to use at each update
		except KeyError: 
			self.kNN_number[self.task_history[-1]] = self.init_kNN_number
		return
			

	def compareTasks(self,data_object,example_inds,new_inds,feature_inds,compute_type='median'): 
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
		if compute_type[0] == 'm':
			result = np.median(costs)
		elif compute_type[0] == 'a':
			result = np.mean(costs)
		return result

	def _determineTaskOrder(self):
		'''utility function to determine the task order to check based on the most recent history of seen tasks'''
		if self.max_task_history_to_consider == -1: 
			task_history_to_consider = self.task_history 
		else: 
			task_history_to_consider = self.task_history[0:min(self.max_task_history_to_consider,len(self.task_history))]
		return utils.getBackwardsUniqueOrder(self.task_history)


class Task(object): 
	def __init__(self, data_object, curr_extrema=[0,10], k=3, basis_dim=2,first_task=False):
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

		####onlineUpdate related
		self.feature_inds = data_object.feature_inds					#features of data_object.all_features that are most relevant to this task

		# if this is the initialization task of the Process object, then the data inds should be the 0th to #frames-in-task indices, else the input data_object will be the "user" object that only has data_vectors added, so there won't be an issue with using the number of vectors in the object as a reference. This could be changed in the future so that the input data_object is always this "user", but for the first task, the user must have already been initialized with all data and the curr_extrema should be changed to be 0:user.num_vectors anyways. 
		if first_task: 
			self.data_inds = np.arange(curr_extrema[1]-curr_extrema[0])
		else:
			v_in_data_object = data_object.num_vectors 
			vectors_added = curr_extrema[1]-curr_extrema[0]
			self.data_inds = np.arange(v_in_data_object-vectors_added,v_in_data_object)		#tasks are only initialized if it has been determined that a new task model is necessary
		self.task_example_inds = self.data_inds	#to be used for comparison to future tasks
		self.curr_labels = []
		
							

		####task related
		self.path = [0,1,2,1,0]				#list of states that the task follows
		self.times = [10,40,100,10,20]		#expected number of frames for the correspondingly indexed path state
		self.states = {}					#dictionary {<state_number>: State instance}
		self.history = []					# all previous data points assigned to this task type 
		self.Taddstate = 0.18				#may want to define this according to how many states exist
		self.basis_dim = basis_dim					#embedding dimension
		self.k = k							#number of clusters
		self.Tdistwidth = 1.5				#start with requiring new points to be within 1.5 standard deviations of each class definition to be considered inside the class
		self.min_state_time = 3				#minimum number of frames a that unique state can half without being absorbed into a nearby state
		self.max_labeled_data_count = 400	#max labled data to test for kNN
		
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
		#print 'Before Simplification: '
		#a = self.printTaskDef()

		#once basic path and times update is settled, merge path segments separated by small amount of secondary tasks
		self.path,self.times = self.simplifyPath(self.path,self.times,self.min_state_time)
		#print 'After simplification: '
		#a = self.printTaskDef()

		#update data_inds
		v_in_data_object = data_object.num_vectors 
		vectors_added = curr_extrema[1]-curr_extrema[0]
		self.data_inds = np.hstack((self.data_inds,np.arange(v_in_data_object-vectors_added,v_in_data_object)))

		#u = self.printTaskDef()
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
		def updatePosition(mixed,path,base_state,new_state,count_new,curr_position,position_threshold = 2): 
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
					# if the count for the new state is acceptable, then if the new state does not exist in the future, shift back to the past and change the necessary time step
					if new_state in path[curr_position:]: 
						curr_position_update_amount = np.argmax(np.array(path[curr_position:])==new_state)
					else: 
						if curr_position > 0: 
							if new_state == path[curr_position-1]:
								curr_position_update_amount = -1
							else:
								curr_position_update_amount = 0

					curr_position += curr_position_update_amount

					#if curr_position == 0: 
					#	curr_position = len(path)-1		#if there is not corresponding position, then default to the last position
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
				future_states_times = np.sum(times[(curr_position+1):])		#the last state in pick and place tasks is typically very short and very quickly completed, so it should be included as a given in the percent complete 
			#print 'Expected future state time remaining: ', future_states_times
			total_task_time = np.sum(np.array(times))		#see details for 'future_state_times'
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
		all_labeled_data = data_object.all_features[self.data_inds,:]	

		#select a random subset of the labeled data [(min(max_labeled_data_count,len(all_labeled_data))) points] to keep speed costs
		max_labeled_data_count = self.max_labeled_data_count
		if len(all_labeled_data) <= max_labeled_data_count: 
			labeled_data = all_labeled_data[:,self.feature_inds]
		else: 
			labeled_data_selection_inds = np.random.permutation(len(all_labeled_data))[0:max_labeled_data_count]
			labeled_data = all_labeled_data[labeled_data_selection_inds,:]
			labeled_data = labeled_data[:,self.feature_inds]

		#pick out the newest data
		curr_data = new_data[self.feature_inds]

		#get initial kNN count
		[knn_label,count_info] = utils.kNN(curr_data,labeled_data,self.labels,k=kNN_number) 
		#print 'Initial Knn:               ', knn_label, count_info 
		#print 'Trouble1: ', curr_frame_count-self.last_state_change_frame
		proportions = mixed.proportionate(curr_frame_count-self.last_state_change_frame)
		#print 'MixedRayleigh proportions: ', proportions

		# incorporate proportions and past few labels
		if proportions == -1: 
			knn_label_updated = knn_label 
		else: 
			count_info_updated = updateCounts(count_info,proportions)
			expectedfrompast = guessFromPast(self.curr_labels)

			# if labels have been added, used expectedfrompast to weight the most likely candidate
			if expectedfrompast != -1: 
				x = count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)]
				x += 1		#this addition and multiplication gives a bit of a chance to low scoring values
				x *= 1.1
				count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)] = x

			# determine new label based on weighted kNN 
			#print 'After guess from past applied: ', count_info_updated
			knn_label_updated = count_info_updated[0][np.argmax(count_info_updated[1])]
		# if curr_frame_count%50 == 0: 
		# 	print 'Init/New label:      '+str(int(knn_label))+'  /  '+str(knn_label_updated)+'\t position: '+str(self.curr_mixed_position)
		self.curr_labels.append(knn_label_updated)

		# update mixedRayleigh distribution for the new frame/possibly new base state 
		new_base_state,new_count_new,new_mixed,new_curr_mixed_position = updatePosition(mixed,self.path,self.base_state,knn_label_updated,self.count_new,self.curr_mixed_position,position_threshold=1)

		# update global variables
		self.base_state = new_base_state
		self.count_new = new_count_new
		# if the mixed position has changed to a more advanced state, update to that state naturally, else update to the new state taking into account how many frames have passed in the already completed states
		if self.curr_mixed_position < new_curr_mixed_position:
			mixed_position_changed = True
			self.last_state_change_frame = curr_frame_count - 3 	#assumes the actual state started just a few frames prior to this new changed mixed rayleigh implementation
			self.curr_mixed_position = new_curr_mixed_position
		elif self.curr_mixed_position > new_curr_mixed_position: 
			mixed_position_changed = True
			self.last_state_change_frame = self.last_state_change_frame-self.frames_since_state_change 	#assumes the previous mixed_position was a fluke and that those states should have actually been given to this new previous mixed_position
			self.curr_mixed_position = new_curr_mixed_position

		# get the percent complete. if 
		self.frames_since_state_change = curr_frame_count-self.last_state_change_frame
		#print 'Frames since state change: ', frames_since_state_change
		self.percent_complete = taskPercentRemaining(self.path,self.times,self.curr_mixed_position,self.frames_since_state_change)
		# if curr_frame_count%50==0:
		# 	print 'Percent complete: \t\t\t'+str(int(self.percent_complete))
		# 	print '---------'

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
		#print 'stateshape: ', G.shape, 'numstates: ', self.num_states
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

	def simplifyPath(self,path,times,min_time=5): 
		'''
		Purpose: 
		Takes in a path (list of ints) with associated times to stay in each index of the path. If a certain path index is less than the threshold, adjust the path to make sense. 
		
		Inputs: 
		path - list of integer states, e.g., [1,2,1,0,1]
		times - list of integer frame counts for each state, e.g., [10,40,15,30,6]
		min_time - integer threshold value for a state to be valid (if the threshold were 7, then the last state in the above example would be fed into the previous state)
		
		Look at the times of each index in the path. If a time is less than the threshold, you should get rid of it. In order to get rid of it, look if it has the same bookends. If so, change the state name to the same as the book ends. Else, assume it is the next state. 

		'''
		path_length = len(path) 
		revamped_path = []
		revamped_times = []
		bad_times = np.array(times)<min_time
		for i,b in enumerate(bad_times): 
			if not b: 
				#if there's no issue with the times value, simply add it
				revamped_path.append(path[i])
				revamped_times.append(times[i])
			else: 
				#otherwise, assume it is the next state in the path unless you are at the last index, when you should assume it is a part of the previous state
				try: 
					revamped_path.append(path[i+1])
					revamped_times.append(times[i])
				except IndexError: 
					revamped_path.append(path[i-1])
					revamped_times.append(times[i])

		#now that the revamped path has be created, merge touching states
		final_path = []
		final_times = []
		k = 0
		end_included = False
		for i,p in enumerate(revamped_path[0:-1]): 
			if k != i: 
				continue
			while k < len(revamped_path)-1 and p == revamped_path[k+1]: 
				if k == len(revamped_path) - 2: 
					end_included = True 
				k+=1
			k += 1
			final_path.append(revamped_path[i])
			final_times.append(sum(revamped_times[i:k]))
		
		if not end_included: 
			final_path.append(revamped_path[-1])
			final_times.append(revamped_times[-1])

		return final_path,final_times

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

