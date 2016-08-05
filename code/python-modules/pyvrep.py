import genio
from kinectData import kinectData
import time
import feature_tools as ft
import numpy as np
import process
import os
import rayleigh
import utils


def main(): 
	global base_state
	global count_new 
	global mixed 
	global curr_mixed_position 
	global last_state_change_frame
	global curr_position 
	global curr_task_labels
	'''
	Structure of tmp.txt:
	---------------------
	line[0]: 	Kinect variable information (joint names, etc)	-CONSTANT
	line[1]: 	Kinect values 									-VREP UPDATED (but always exists)
	line[2]: 	('*' (curr task complete) or '')				-VREP CONTROLS '*' placement 
	line[3]: 	(curr_state, percent_complete, etc)				-PYTHON UPDATED
	line[4]: 	('COMPLETE' (process ended))					-VREP UPDATED

	tmp.txt change flow: 
	---------------------
	1. VREP sends >5 line file with first full task of data 
	2. PYTHON updates tmp.txt to 4 lines with relevant details on line[3]
	3a. VREP updates line[1],line[2] with new values - tmp.txt becomes either 2 or 3 lines long
		-- repeat from step 2 for additional data 
	3b. if no more lines to be given for processing, VREP updates tmp.txt to be 5 lines long with line[4] marking the end of process

	'''
	'''
	Pseudocode 

	Initialization 
	1. Create a kinectData object once first full task has been seen
	2. add all data to kinectData object at once for first task
	3. get features from data
	4. define the initial task process and state-transition-path

	After an initial task has been defined 
	1. Read tmp.txt for line count
	2a. if linecount == 2 or == 3
		- add tmp.txt to kinectData (remove line 3 if necessary), 
		if 2: (task is still ongoing)
			- get percent_complete, current state, current task, task probability
			- append variables to tmp.txt on fourth line ('\n\ninformation')
		if 3: (task has completed)
			- get information = percent_complete, current state, current task, task probability
			- append variables to tmp.txt on fourth line ('\ninformation')
			- update task using full task data (create new task if probability is low enough)
			- 
	2b. elif linecount == 4 
		- pass
	2c. elif linecount == 5 
		- stop running and wait
	'''
	def initialize(ID): 
		
		def getMidpoint():
			dummy1 = kinectData('4')
			dummy2 = kinectData('5')
			dummy1.addData(midptfile)
			dummy2.addData(midptfile)
			names_base, dummy1xyz = ft.disectJoints(dummy1.names_list,np.array(dummy1.data_array))
			names_base, dummy2xyz = ft.disectJoints(dummy2.names_list,np.array(dummy2.data_array))
			midpt = ft.getMidpointXYZ(dummy1xyz,dummy2xyz)
			return midpt

		data = kinectData(ID)
		data.addData(tmpfile)
		data.similarity_method = 'exp'
		data.midpoint = getMidpoint()
		data.getFeatures()
		print 'num vectors: ', data.num_vectors
		task = process.Task(data,[0,data.num_vectors])
		x = task.printTaskDef()
		#potentially add process here....once I get to it...
		return data,task

		#do initialization of kinectData object and task stuff

	def onlineUpdate(complete=False): 
		# define used global variables
		global base_state
		global count_new 
		global mixed 
		global curr_mixed_position 
		global last_state_change_frame
		global curr_position 
		global curr_task_labels
		#global task 
		#global data 

		def updateCounts(count_info,proportions): 
			#print 'PREupdate: ', count_info[0],count_info[1], '* ', proportions
			for ind,state in enumerate(count_info[0]): 
				count_info[1][ind] = float(count_info[1][ind]) * float(proportions[state])
			#print 'POSTupdate: ', count_info[0],count_info[1]
			return count_info

		def updatePosition(mixed,base_state,new_state,count_new,curr_position,position_threshold = 4): 
			'''
			pseudocode: 
			if base_state == new_state, return count_new = 0, mixed unaltered
			else count_new++, if count_new > position_threshold, increment position and recreate mixed, count_new = 0, base_state = new_state, else return 
			return count_new 
			'''
			print 'Current rayleigh position: ', curr_position
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
					base_state = new_state
			print 'New rayleigh position: ', curr_position, 'base state new: ', base_state
			print task.path, task.times
			return base_state, count_new, mixed, curr_position

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
				print 'Guess from past function consider: ', consider
				best,aux = utils.majorityVote(consider)
				return best
			else: 
				return -1

		def taskPercentRemaining(task,curr_position,differential): 
			curr_state_time_remaining = max(task.times[curr_position]-differential, 0)
			print 'task.times[curr_position]-differential: ', task.times[curr_position], '-', differential,'=', task.times[curr_position]-differential
			print 'Current state time remaining: ', curr_state_time_remaining
			if curr_position == len(task.path)-1: 
				future_states_times = 0
			else: 
				future_states_times = np.sum(task.times[(curr_position+1):])
			print 'Expected future state time remaining: ', future_states_times
			total_task_time = np.sum(task.times)
			percent_remaining = 100*(curr_state_time_remaining+future_states_times)/float(total_task_time)
			percent_complete = 100 - percent_remaining
			return percent_complete
			


		# get labeled data to use for comparison
		kNN_number = 20		#look at 20 nearest neighbors

		# add the new line of data and get features
		data.addData(tmpfile)
		data.getFeatures()
		# separate labeled data from new data
		new_data = data.feat_array[data.num_vectors-1,:]
		print 'New data considered: ', new_data

		# some local variables for determining current frame and expected frame along the state-transition-path
		frames_in_curr_task = data.num_vectors - last_task_end		#frames since last task ended
		
		# get init guess at knn_label and probability proportions based on current position within the task
		[knn_label,count_info] = utils.kNN(new_data,labeled_data,task.labels,k=kNN_number) 
		print 'Initial Knn: ', knn_label, count_info 
		proportions = mixed.proportionate(frames_in_curr_task-last_state_change_frame)
		print 'MixedRayleigh proportions: ', proportions

		# incorporate proportions and past few labels
		count_info_updated = updateCounts(count_info,proportions)
		print 'After proportions considerations: ', count_info_updated	
		expectedfrompast = guessFromPast(curr_task_labels)
		print 'Guess from past: ', expectedfrompast

		# if labels have been added, used expectedfrompast to weight the most likely candidate
		if expectedfrompast != -1: 
			x = count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)]
			x += 3
			x *= 1.1
			count_info_updated[1][np.argmax(np.array(count_info_updated[0])==expectedfrompast)] = x

		# determine new label based on weighted kNN 
		print 'After guess from past applied: ', count_info_updated
		knn_label_updated = count_info_updated[0][np.argmax(count_info_updated[1])]
		print 'New knn label chosen: ', knn_label_updated
		curr_task_labels.append(knn_label_updated)

		# update mixedRayleigh distribution for the new frame/possibly new base state 
		new_base_state,new_count_new,new_mixed,new_curr_mixed_position = updatePosition(mixed,base_state,knn_label_updated,count_new,curr_mixed_position,position_threshold=4)

		# update global variables

		base_state = new_base_state
		count_new = new_count_new
		mixed = new_mixed
		if curr_mixed_position != new_curr_mixed_position:
			last_state_change_frame = frames_in_curr_task
			curr_mixed_position = new_curr_mixed_position

		frames_since_state_change = frames_in_curr_task-last_state_change_frame
		print 'Frames since state change: ', frames_since_state_change
		percent_complete = taskPercentRemaining(task,curr_mixed_position,frames_since_state_change)

		# define information string
		task_id = 0		#not implemented to determine which task is happening, so just making it task 0 for now....
		information = str(curr_task_labels[-1])+'\t'+str(percent_complete)+'\t'+str(task_id)
		print 'information: '+information
		return information

	def taskUpdate(): 
		pass
		#update task information 

	def setupFiles():
		#setup tmp file if it does not exist
		if os.path.isfile(tmpfile): 		#possibly do something different if file already exists...
			with open(tmpfile,'w') as f: 
				f.write('1\n2\n3\n4')		
		else: 
			with open(tmpfile,'w') as f: 
				f.write('1\n2\n3\n4')

		#setup starts file
		if os.path.isfile(startsfile):		#possibly do somethin different if file already exists, like do the full analysis for the file
			starts = [224, 327, 474, 678, 807, 909, 995, 1168, 1268, 1367, 1481, 1580]	#values from p2-3.txt result
			starts_str = [str(x) for x in starts]
			starts_str = '\t'.join(starts_str)
			with open(startsfile,'w') as f: 
				f.write(starts_str)
		else: 
			starts = [224, 327, 474, 678, 807, 909, 995, 1168, 1268, 1367, 1481, 1580]	#values from p2-3.txt result
			starts_str = [str(x) for x in starts]
			starts_str = '\t'.join(starts_str)
			with open(startsfile,'w') as f: 
				f.write(starts_str)

		#setup midpoint file
		if os.path.isfile(midptfile): 		#possibly do something different if file already exists...
			with open(midptfile,'w') as f: 
				f.write('empty')		
		else: 
			with open(midptfile,'w') as f: 
				f.write('empty')

	def sendTmpResponse(information): 
		'''information is a single string that will be placed on line 4 of tmpfile'''
		genio.shortenfile(tmpfile,lines,2)
		genio.appendline(tmpfile,information)		#no lineshift necessary to add to information because tmpfile line 2 already has one \n at the end, appendline adds another \n, thus putting information on line 4

	'''necessary initialization variables'''
	tmpfile = '/home/josh/Desktop/tmp.txt'		#file name of tmp file for communication between vrep and python
	startsfile = '/home/josh/Desktop/starts.txt'	#file name of handover starts data 
	midptfile = '/home/josh/Desktop/midpoint.txt'	#file name of midpoint information (acquired by Vrep)
	lineshift = '\n'		#for use in appending information to the fourth line of a currently two line file
	userID = '4'

	setupFiles()		#makes sure tmpfile and startsfile exist and are prepped

	running = True		#enter the loop
	thing = 0
	initialized = False

	while(running): 
		#initialization routine (wait for vrep to start)
		#check access to tmpfile
		#time.sleep(0.001)
		if os.path.exists(tmpfile): 
			try: 
				os.rename(tmpfile,tmpfile)
				#print 'access granted'
				lines,count = genio.getlines(tmpfile)
			except OSError as e: 
				print 'ERRROR'+str(e)
				continue


		if initialized == False: 
			#check if full first task complete (VREP will only send first chunk of data with many lines in it)
			starts_count = genio.linecount(startsfile)		#vrep adds second line to starts file if first task complete
			print 'curr count: ',count, 'startscount: ', starts_count
			if starts_count == 2: 	
				print 'initializing'
				data,task = initialize(userID) 
				initialized = True	
				#running = False
				information = str(task.path[0])+'\t0.0\t0'			# 'curr_state	percent_complete	curr_task'
				sendTmpResponse(information)
				
				#initialization stuff for next task process
				curr_task_labels = []								# labels for the current task 
				last_task_end = data.num_vectors					# last frame to consider as labeled data
				labeled_data = data.feat_array[0:last_task_end,:]
				base_state = task.path[0]							# based state for the mixed Rayleigh distribution used in onlineUpdate()
				count_new = 0										# variable for counting unexpected states showing up in kNN process considering the expected state from the state-transition-path
				last_state_change_frame = 0							#holder for state-transition points
				# setup mixed Rayleigh
				curr_mixed_position = 0
				mixed = rayleigh.MixedRayleigh(task, position=curr_mixed_position)
				print 'mixed initialized.'

		else: 
			if count == 2: 				#new online data exists
				'''get information for progress and current state'''
				print '\n\ncurr_task_labels: ', curr_task_labels
				information = onlineUpdate() 
				print 'numvectors post update: ', data.num_vectors
				sendTmpResponse(lineshift+information)
				#running = False
				if data.num_vectors > 500: 
					return data,task

			elif count == 3: 			#task iteration is complete
				'''update task definition / define a new task'''
				information = onlineUpdate(complete=True)	#make sure to send back 100% complete
				sendTmpResponse(information)
				taskUpdate()

				#initialization stuff for next task process
				curr_task_labels = []	#reset the current task labels 
				last_task_end = data.num_vectors 
				labeled_data = data.feat_array[0:last_task_end,:]
				base_state = task.path[0]
				count_new = 0
				last_state_change_frame = 0							#holder for state-transition points
				# setup mixed Rayleigh
				curr_mixed_position = 0
				mixed = rayleigh.MixedRayleigh(task, position=curr_mixed_position)

			elif count == 4: 			#no update from vrep yet
				pass #(or wait some short amount of time before attempting to process again)

			elif count == 5: 			#process is complete, stop running 
				running = False















if __name__=='__main__': main()