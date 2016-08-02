import genio
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
	def initialize(ID): 
		def getMidpoint():
			with open(midptfile,'r') as f: 
				midpt = [float(k) for k in f.read().split('\n')]
			data.midpoint[1,0] = midpt[0]
			data.midpoint[1,1] = midpt[1]
			data.midpoint[1,2] = midpt[2]

		data = kinectData(ID)
		data.addData(tmpfile)
		getMidpoint()

		task = process.Task(data,[0,data.num_vectors])
		#potentially add process here....once I get to it...
		return data,task

		#do initialization of kinectData object and task stuff

	def onlineUpdate(complete=False): 
		pass
		#get current state and what not

	def taskUpdate(): 
		pass
		#update task information 

	def setupFiles():
		#setup tmp file if it does not exist
		import os.path 
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
			with open(tmpfile,'w') as f: 
				f.write('empty')		
		else: 
			with open(tmpfile,'w') as f: 
				f.write('empty')



	tmpfile = '/home/josh/Desktop/tmp.txt'		#file name of tmp file for communication between vrep and python
	startsfile = '/home/josh/Desktop/starts.txt'	#file name of handover starts data 
	midptfile = '/home/josh/Desktop/midpoint.txt'	#file name of midpoint information (acquired by Vrep)
	lineshift = '\n\n'		#for use in appending information to the fourth line of a currently two line file
	userID = '4'

	setupFiles()		#makes sure tmpfile and startsfile exist and are prepped

	running = True		#enter the loop
	thing = 0
	initialized = False

	while(running): 
		#initialization routine (wait for vrep to start)
		lines,count = genio.getlines(tmpfile)


		'''test code begin'''
		if count == 2: 
			print 'code_accepted, data to be added'
		elif count == 3: 
			print 'code accepted, data to be added, tasks to be updated'
		elif count == 4: 
			print 'PASS, no changes have occured'
		elif count == 5: 
			print 'end process, file over'
			running = False

		thing += 1
		if thing%5 == 0: 
			genio.appendline(tmpfile,'appendthis')


		'''test code end'''






		if not initialized: 
			#check if full first task complete (VREP will only send first chunk of data with many lines in it)
			starts_count = genio.linecount(startsfile)		#vrep adds second line to starts file if first task complete
			if starts_count == 2: 	
				print 'initializing'
				data,task = initialize(userID) 
				initialized == True	
		else: 
			if count == 2: 				#new online data exists
				'''get information for progress and current state'''
				information = onlineUpdate() 
				genio.appendline(tmpfile,lineshift+information)

			elif count == 3: 			#task iteration is complete
				genio.shortenfile(tmpfile,lines,2)	#tmp.txt now has the proper form to be input to kinectData.addData(tmpfile)
				'''update task definition / define a new task'''
				information = onlineUpdate(complete=True)	#make sure to send back 100% complete
				genio.appendline(tmpfile,lineshift+information)
				taskUpdate()

			elif count == 4: 			#no update from vrep yet
				pass #(or wait some short amount of time before attempting to process again)

			elif count == 5: 			#process is complete, stop running 
				running = False















if __name__=='__main__': main()