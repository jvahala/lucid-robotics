'''main.py performs all calculations with regard to incoming data'''

'''
Key words: 
Process - the full set of tasks for a single study. One process is comprised of one or more tasks. The tasks could be the same or could vary. 
Task - The set of data between endpoints defined by the actions themselves. For instance, a task may be defined by a robot's release of a plate to a user. What happens in between these endpoints may be different between tasks. 
State - One of a set of actions that are followed in performing a task. Each unique task follows a known state-transition-path which is the path of states that occur from task endpoint to endpoint. 
'''

#1. Initialize a process to hold incoming tasks (current_task = 0)

#2. Once a task is known to begin, start gathering data into a kinectData object data_array

#2a. Send robot on proactive trajectory to start next task whenever the user is ready 

#once new tasks begins
#3a. Determine key features from task

#3b. Initialize a task object to get state_transition_path
#3bi. add task to process (increment current_task to 1)

#for each incoming frame, 
#3c. Add frame to data_array

#3d. Calculate features for frame and add to feat_array

#3e. For each known task, determine the current state for the relevant features through KNN matching of current features to features previously matching each state in the state-transition-path

#3f. Calculate probability that each known task is begin performed

#3g. Calculate expected time to completion of each known task

#3hi. Calculate expected task completion time by summing expected times to completion weighted by task probability
#3hii. Calculate similarly weighted robot progression metric 

#3i. Send robot progression metric to robot (This should be all the robot needs to act) 

#once new task is complete
#4. Determine if new task was novel, YES -> 3a, NO -> 3b (update task model for most likely task)
from kinectData import kinectData 
import handover_tools as ht 
import feature_tools as ft
from process import Process,Task,State
import utils
import rayleigh
import numpy as np

# def label(data_object, feature_range=[0,0], num_clusters=3, basis_dim=2, k_nn=6):
# 	input_data = data_object.feat_array[feature_range[0]:feature_range[1],:]
# 	labels, centers, U = clusters.spectralClustering(input_data, similarity_method=data_object.similarity_method, k_nn=k_nn, basis_dim=basis_dim, num_clusters=num_clusters)
# 	labels = utils.orderStates(labels)
# 	labels = [int(x) for x in labels]
# 	labels = list(labels)
# 	print 'labels: \n', labels,'\nCenters: \n', centers, '\nnorm val: ', data_object.norm_value, '\nframes: ', feature_range[0] ,'to ', feature_range[1]
# 	return labels, centers, U

def fileSetup(file_name,user1_id,user2_id): 
	def setup(user_id='4', file_name=''): 
		data = kinectData(user_id) 				#set up kinectData object with user_id 
		data.addData(file_name)					#adds all data in file with correct user_id
		data.similarity_method = 'exp'			#uses exponential similarity array 
		data.getFeatures()						#gets the features for all elements in the file and stores them in data.feat_array
		return data

	data1 = setup(user1_id,file_name)
	data2 = setup(user2_id,file_name)
	data1.midpoint = ft.getMidpointXYZ(data1.dataXYZ[0,:,:],data2.dataXYZ[0,:,:])
	data2.midpoint = data1.midpoint
	handover_starts = ht.handoverID(data1,data2)
	return data1,data2,handover_starts

def begin(): 
	def getFilename(taskid='p2-3',pc='mac'):
		if pc == 'mac':  
			base_name = '/Users/vahala/Desktop/Wisc/LUCID/Handover-Data/p$/p$-logs/#.txt'
			print 'using MAC file address'
		elif pc == 'linux':
			base_name = '/home/josh/Desktop/V-REP_PRO_EDU/#.txt'
			print 'using LINUX file address'
		fid = taskid[1]
		full_name = base_name.replace('$',fid).replace('#',taskid)
		return full_name


	#setup file name information
	task_id = 'p2-3'				#task from the dataset 
	file_name = getFilename(task_id,pc='linux')
	rec_id,giv_id = '4','5'			#receiver and giver id numbers from the dataset

	#set up data objects from full file with proper midpoint
	receiver,giver,starts = fileSetup(file_name,rec_id,giv_id)

	#define initial task using spectral clustering
	task = Task(data_object=receiver, curr_extrema=[starts[0],starts[1]], k=3)

	#set up a process with one task added and increment current task from 0 to 1
	proc = Process()		#need to implement a Process.addTask() method 
	#proc.addTask(task)

	return receiver,giver,starts,task

def iterateTask(curr_task_id,receiver,starts,task,testvalue): 
	'''for use after main.begin has been called to get receiver,giver,starts,task,curr_labels'''
	def updateCounts(count_info,proportions): 
		#print 'PREupdate: ', count_info[0],count_info[1], '* ', proportions
		for ind,state in enumerate(count_info[0]): 
			count_info[1][ind] = count_info[1][ind] * proportions[state]
		#print 'POSTupdate: ', count_info[0],count_info[1]
		return count_info

	def updatePosition(mixed,base_state,new_state,count_new,curr_position,position_threshold = 4): 
		'''
		pseudocode: 
		if base_state == new_state, return count_new = 0, mixed unaltered
		else count_new++, if count_new > position_threshold, increment position and recreate mixed, count_new = 0, base_state = new_state, else return 
		return count_new 
		'''
		if base_state == new_state: 
			count_new = 0 
		else: 
			count_new += 1
			if count_new > position_threshold: 
				print 'curr/next = ', curr_position, '/',np.argmax(np.array(task.path[curr_position:])==new_state), 'new: ', new_state
				curr_position += np.argmax(np.array(task.path[curr_position:])==new_state)
				print curr_position, 'llllll'
				if curr_position == 0: 
					curr_position = len(task.path)-1		#if there is not corresponding position, then default to the last position
				mixed = rayleigh.MixedRayleigh(task, curr_position)		#update the mixedRayleigh
				k = i-position_threshold
				base_state = new_state
		return base_state, count_new, mixed, curr_position

	def guessFromPast(curr_labels,past_length=3):
		'''
		gets majority vote from past maximum 'past_length' number of labels (or all existing labels) in curr_labels 
		'''
		numlabels = len(curr_labels)
		if numlabels>0:
			if numlabels > past_length:
				consider = curr_labels[-past_length]
			else: 
				consider = curr_labels 
			best,aux = utils.majorityVote(consider)
			return best
		else: 
			return -1

	def taskPercentRemaining(task,curr_position,differential): 
		curr_state_time_remaining = max(task.times[curr_position]-differential, 0)
		if curr_position == len(task.path)-1: 
			future_states_times = 0
		else: 
			future_states_times = np.sum(task.times[(curr_position+1):])
		total_task_time = np.sum(task.times)
		percent_complete = 100*(curr_state_time_remaining+future_states_times)/float(total_task_time)
		percent_remaining = 100 - percent_complete
		return percent_remaining

	curr_labels = []
	old = receiver.feat_array[starts[0]:starts[curr_task_id]]
	consider = receiver.feat_array[starts[curr_task_id]:starts[curr_task_id+1]]
	curr_position = 0
	mixed = rayleigh.MixedRayleigh(task, position=curr_position)
	i = 0
	k = 0
	for frame in consider:
		if i == 100: 
			pass
		if i == 0: 
			curr_labels.append(task.path[0])
			base_state = task.path[0]
			count_new = 0
			i += 1
			continue
		#need to check if mixedRayleigh needs to be updated to the new position, so if the initial state value changes and stays like that for n iterations, then update the position
		kNNnumber = 20
		[knn_label,count_info] = utils.kNN(frame,receiver.feat_array[starts[0]:starts[curr_task_id]],task.labels, k=kNNnumber) 
		print 'mixed position: ', mixed.position
		proportions = mixed.proportionate(i-k)
		#choose label by adding in proportions to consideration
		count_info_updated = updateCounts(count_info,proportions)	
		#incorporate smoothing by further weighting the past few states
		expectedfrompast = guessFromPast(curr_labels)
		'''if i == 5: 
			print '1: ', np.argmax(np.array(count_info_updated[0])==expectedfrompast)
			print '2: ', expectedfrompast
			print '3; ', count_info_updated 
			print '4: ', count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)]
			break '''
		if expectedfrompast != -1: 
			x = count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)]
			x += 3
			x *= testvalue
			count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)] = x
		knn_label_updated = count_info_updated[0][np.argmax(count_info_updated[1])]
		curr_labels.append(knn_label_updated)

		new_base_state, new_count_new, new_mixed, new_curr_position= updatePosition(mixed, base_state,knn_label_updated,count_new,curr_position,position_threshold=4)

		base_state = new_base_state
		count_new = new_count_new
		mixed = new_mixed
		if curr_position != new_curr_position:
			k = i
			curr_position = new_curr_position

		percent_complete = taskPercentRemaining(task, curr_position, i-k)


		'''uncomment the following two lines to print the kNN count info associated with each frame'''
		print '\n\nframe number: ', i, i-k
		print count_info_updated
		print 'percent complete: ', percent_complete

		i += 1
	return [int(x) for x in curr_labels]

if __name__=='__main__': begin()







