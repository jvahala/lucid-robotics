from genio import appendline, getlines, linecount, shortenfile
from kinectData import kinectData

def main(): 
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
	def initialize(): 
		#do initialization of kinectData object and task stuff

	def onlineUpdate(): 
		#get current state and what not

	def taskUpdate(): 
		#update task information 

	tmpfile = 'tmp.txt'
	lineshift = '\n\n'		#for use in appending information to the fourth line of a currently two line file

	while(running): 
		#initialization routine (wait for vrep to start)
		lines,count = getlines(tmpfile)
		if not initialized: 
			#check if full first task complete (VREP will only send first chunk of data with many lines in it)
			if count > 5:
				data,task = initialize() 
		else: 
			if count == 2: 				#new online data exists
				'''get information for progress and current state'''
				information = onlineUpdate() 
				appendline(tmpfile,lineshift+information)

			elif count == 3: 			#task iteration is complete
				shortenfile(tmpfile,lines,2)	#tmp.txt now has the proper form to be input to kinectData.addData(tmpfile)
				'''update task definition / define a new task'''
				information = onlineUpdate(complete=True)	#make sure to send back 100% complete
				appendline(tmpfile,lineshift+information)
				taskUpdate()

			elif count == 4: 			#no update from vrep yet
				pass #(or wait some short amount of time before attempting to process again)

			elif count == 5: 			#process is complete, stop running 
				running = False















if __name__=='__main__': main()