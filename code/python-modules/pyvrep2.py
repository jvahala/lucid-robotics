import genio
from kinectData import kinectData
#import time
import feature_tools as ft
import numpy as np
import process
import os
import rayleigh
import utils


def main(): 
	#global base_state
	#global count_new 
	#global mixed 
	#global curr_mixed_position 
	#global last_state_change_frame
	#global curr_position 
	#global curr_task_labels
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
		evolution = process.Process(task)
		return data,task,evolution

		#do initialization of kinectData object and task stuff

	def onlineUpdate(kNN_number=20,complete=False): 
		# define used global variables
		#global base_state
		#global count_new 
		#global mixed 
		#global curr_mixed_position 
		#global last_state_change_frame
		#global curr_position 
		#global curr_task_labels
		#global task 
		#global data 
			
		### collect the newest data point and add to kinectData object 
		# add the new line of data and get features
		data.addData(tmpfile)
		data.getFeatures()
		# separate labeled data from new data
		new_data = data.all_features[data.num_vectors-1,:]

		### perform online update
		percent_complete = evolution.onlineUpdate(new_data,data)
		if complete: 
			percent_complete = 100.0

		### define information string
		#determine the current most likely state
		if len(evolution.known_tasks[0].curr_labels) > evolution.min_frames_for_probability: 
			temp_value = 0
			for t_id,value in evolution.task_online_probability.iteritems(): 
				if value > temp_value: 
					task_id = t_id 
					temp_value = value
		else: 
			task_id = evolution.task_history[-1]

		curr_label = evolution.known_tasks[task_id].curr_labels[-1]

		information = str(curr_label)+'\t'+str(percent_complete)+'\t'+str(task_id)
		print 'information: '+information
		return information, percent_complete

	def processUpdate(): 
		### perform update to the proper task
		evolution.updateKnownTasks(data)

		### create string response of current state for vrep
		task_id = evolution.task_history[-1]
		curr_label = evolution.known_tasks[task_id].curr_labels[-1]
		percent_complete = 0.0
		new_information = str(curr_label)+'\t'+str(percent_complete)+'\t'+str(task_id)
		return new_information

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
	''''''

	'''user set variables'''
	kNN_number = 20		#look at 20 nearest neighbors
	percent_complete_threshold = 80.0		#tell arm to move to 100% state if percent complete threshold goes above 80%
	''''''

	#program start
	setupFiles()		#makes sure tmpfile and startsfile exist and are prepped

	thing = 0
	initialized = False
	running = True		#enter the loop

	while(running): 
		#initialization routine (wait for vrep to start)
		#check access to tmpfile
		#time.sleep(0.001)				#for use if there are issues with file opening

		#check for issues accessing files, if issues exist, try again until they don't
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
				data,task,evolution = initialize(userID) 		
				initialized = True	
				#running = False
				information = str(task.path[0])+'\t0.0\t0'			# 'curr_state	percent_complete	curr_task'
				sendTmpResponse(information)
				
				#initialization stuff for next task process
				#curr_task_labels = []								# labels for the current task 
				#last_task_end = data.num_vectors					# last frame to consider as labeled data
				#labeled_data = data.feat_array[0:last_task_end,:]
				#base_state = task.path[0]							# based state for the mixed Rayleigh distribution used in onlineUpdate()
				#count_new = 0										# variable for counting unexpected states showing up in kNN process considering the expected state from the state-transition-path
				#last_state_change_frame = 0							#holder for state-transition points
				task_is_complete = False
				# setup mixed Rayleigh
				#curr_mixed_position = 0
				#mixed = rayleigh.MixedRayleigh(task, position=curr_mixed_position)

		else: 
			if count == 2: 				#new online data exists
				'''get information for progress and current state'''
				#print '\n\ncurr_task_labels: ', curr_task_labels
				information, percent_complete = onlineUpdate(kNN_number=kNN_number, complete=task_is_complete) 
				if percent_complete > percent_complete_threshold: 
					task_is_complete = True
				#print 'numvectors post update: ', data.num_vectors
				sendTmpResponse(lineshift+information)
				#running = False
				if data.num_vectors > 1500: 
					return data,evolution
				threepeat = False

			elif count == 3: 			#task iteration is complete
				#threepeat is meant for dealing with timing issues where vrep is bieng slow to process the new file
				if threepeat == True: 
					pass
				else: 
					genio.shortenfile(tmpfile,lines,2)
					#task_start = data.num_vectors - len(curr_task_labels) - 1
					#task_end = data.num_vectors
					print '\n\nOnlineUpdate:'
					information, percent_complete = onlineUpdate(kNN_number=kNN_number, complete=task_is_complete)	#make sure to send back 100% complete
					sendTmpResponse(information)
					print '\n\nTaskUpdate: '
					new_information = processUpdate()
					print '------------------------------------------'
					#running = False

					#initialization stuff for next task process
					#curr_task_labels = []	#reset the current task labels 
					#last_task_end = data.num_vectors 
					#labeled_data = data.feat_array[0:last_task_end,:]
					#base_state = task.path[0]
					#count_new = 0
					#last_state_change_frame = 0							#holder for state-transition points
					task_is_complete = False
					# setup mixed Rayleigh
					#curr_mixed_position = 0
					#mixed = rayleigh.MixedRayleigh(task, position=curr_mixed_position)
					#threepeat = True
					#task_id = 0
					sendTmpResponse(new_information)
					#print 'whops'

			elif count == 4: 			#no update from vrep yet
				pass #(or wait some short amount of time before attempting to process again)

			elif count == 5: 			#process is complete, stop running 
				running = False
				return data,evolution















if __name__=='__main__': main()