#import multi_test
from kinectData import kinectData
import numpy as np
import handover_tools as ht
import process
import matplotlib.pyplot as plt
from copy import deepcopy

import time

np.set_printoptions(threshold=np.nan,precision=2,suppress=True)

def getFileDict(filepath,txtname,fileDict=None): 
	'''
	returns a dictionary indexed by the txtname of the file (from txtname.txt....ie the .txt is left out of txtname)
	'''
	if fileDict == None: 
		fileDict = {}
	fileDict[txtname] = filepath+txtname+'.txt'
	return fileDict

def setup(): 
	#define filenames that contain txt files
	filepath = "/Users/vahala/Desktop/Wisc/LUCID/josh-data/"
	txtnames = ['cabinet_fast', 'counter_fast', 'floor_fast', 'middle_sit', 'phone_counter_fast', 'cabinet_slow', 'counter_slow', 'floor_slow', 'onepushup', 'phone_counter_slow']
	default_midpoint = np.array([-0.5,0.0,2.0])		#found from plotting x,y,z of various examples (the key is that a stationary point exists that we can get features from for movement within the environment)

	fileDict = {}
	for _txtname in txtnames: 
		fileDict = getFileDict(filepath,_txtname,fileDict)

	#add data to individual objects associated with each txtname - these are dummy data holders essentially
	kinectDict,startsDict = {},{}
	for _txtname,path in fileDict.iteritems(): 
		print 'txtname: ', _txtname
		new_data = kinectData('1')
		new_data.addData(path)
		new_data.similarity_method = 'exp'
		new_data.midpoint = default_midpoint

		new_data.getFeatures()
		kinectDict[_txtname] = new_data 
		startsDict[_txtname] = ht.handoverIDSingle(new_data)

	print kinectDict
	'''
	import genio 
	genio.writeFileFromDictionary('/Users/vahala/Desktop/TESTING.txt',startsDict)
	'''

	return txtnames,kinectDict,startsDict

def initializeUser(idnum,init_object,init_starts):
	#create an object
	user = kinectData(idnum)

	#initialize data array and necessary variables for getting features
	user.data_array = init_object.data_array[init_starts[0]:init_starts[1],:]
	user.num_vectors = len(user.data_array)
	user.names_list = init_object.names_list	
	user.midpoint = init_object.midpoint
	user.similarity_method = init_object.similarity_method

	#get features
	user.getFeatures(exp_weighting=True)

	return user

def getNewFeatures(user,data): 
	#update the data array
	user.data_array = np.vstack((user.data_array,data))
	user.num_vectors += 1

	#get the new features
	user.getFeatures()
	new_features = user.all_features[-1,:]

	return new_features


def testRun(kinectDict,startsDict,txtnames,tasktypes_ordered,startiter_ordered,numtasks_ordered,_plot=True):
	# go through the ordered taskIDs starting at the relevant starttask number and iterate for the relevant number of tasks
	'''
	ie if tasktypes_ordered = [1,3,4], startiter_ordered = [3,1,0], and numtasks_ordered = [4,2,2]: 
	- start with taskID number 1 at the index 3 iteration (the fourth iteration), and go through 4 task iterations before switching to taskID number 3 starting at index 1 for 2 iterations and so on. 
	'''
	initialized = False		#counter for number of tasks that have so far been completed
	pct_complete_list = []
	timings_online = []
	timings_update = []


	for i,tasktype in enumerate(tasktypes_ordered): 

		#gather the relevant data inds for this tasktype 
		print '\nTaskType: ', txtnames[tasktype]
		curr_txtname = txtnames[tasktype]
		curr_data_obj = kinectDict[curr_txtname]
		curr_starts = startsDict[curr_txtname]
		relevant_starts = curr_starts[startiter_ordered[i]:startiter_ordered[i]+numtasks_ordered[i]+1]	#all the relevant start indices
		starts_shift = relevant_starts[0]
		iters_left = numtasks_ordered[i]		#number of this tasks iterations left to be completed
		curr_task_num = 0

		#timing information 

		
		

		#iterate through the data inds 
		#if no initial task has been completed, complete one and create a Process object
		if not initialized: 
			print 'Initializing'
			init_task = process.Task(curr_data_obj, curr_extrema=[relevant_starts[0],relevant_starts[1]], k=3, basis_dim=2, first_task=True)
			print 'init data_inds', init_task.data_inds
			evolution = process.Process(init_task)
			user = initializeUser('1',curr_data_obj,[relevant_starts[0],relevant_starts[1]])
			iters_left -= 1			#one less iteration is left to be completed
			curr_task_num += 1
			initialized = True 
		
		# go through each frame in relevant data, if frame number equals 
		print len(user.data_array)
		while iters_left > 0: 
			start = time.clock()
			#for plotting things on each run through
			errors_plot_list, error_metric_plot_list, probability_plot_list, pct_plot_list, est_pct_plot_list = [],[],[],[],[]
			#online process
			curr_task_data_inds = np.arange(relevant_starts[curr_task_num],relevant_starts[curr_task_num+1])
			curr_task_data = curr_data_obj.data_array[curr_task_data_inds,:]
			for data in curr_task_data: 
				#collect new features and perform online update process
				new_features = getNewFeatures(user,data)
				pct_complete = evolution.onlineUpdate(new_features,user)

				#interesting plotting stuff
				if _plot: 
					errors_plot_list.append(deepcopy(evolution.error_per_frame))
					error_metric_plot_list.append(deepcopy(evolution.error_metric))
					probability_plot_list.append(deepcopy(evolution.task_online_probability))
					pct_plot_list.append(deepcopy(evolution.task_pct_complete))
					est_pct_plot_list.append(pct_complete)
				#-------------------------

			end = time.clock()
			timings_online.append((end-start)/float(evolution.curr_frame_count))

			errors_per_frame = {}
			for k,v in evolution.cumulative_online_errors.iteritems():
				errors_per_frame[k] = max(1,max(1, (v/float(evolution.curr_frame_count))-evolution.known_task_count)*evolution.known_task_count - evolution.known_task_count)
			
			'''plot interesting plot stuff
			'''
			if _plot: 
				fig = plt.figure()
				nameit = txtnames[tasktype]
				plt.suptitle(nameit)
				plotInformation(fig,errors_plot_list,221,'Error per Frame','Error Value',maxy=104)
				plotInformation(fig,error_metric_plot_list,222,'Error Metric','Metric Value',maxy=304)
				plotInformation(fig,probability_plot_list,223,'Task Probabilities','Probability',maxy=1.09)
				plotInformation(fig,pct_plot_list,224,'Percent Complete','Percent',maxy=109)
				plotInformation(fig,est_pct_plot_list,224,'Percent Complete','Percent',maxy=109,dict_input=False)
				fig.tight_layout()

			

			print '\ntask online probabilities: ', evolution.task_online_probability, '\tonline errors: ', errors_per_frame
			print evolution.task_pct_complete, 'Probable pct complete: ', pct_complete
			pct_complete_list.append(pct_complete)


			#taskupdate process
			start = time.clock()
			evolution.updateKnownTasks(user,compute_type='average')
			end = time.clock()
			timings_update.append(end-start)
			#setup for next task
			curr_task_num += 1 		#move to next task number (will only compute it if there iters_left > 0, so no worries)
			iters_left -= 1
			pct_complete_list = [int(x) for x in pct_complete_list]

	add_timings = list(np.array(timings_online)+np.array(timings_update))
	print 'online timings (sec per frame): ', timings_online
	print 'update timings (sec): ', timings_update
	print 'timings total (sec): ', add_timings

	print 'online Hz: ', [1/x for x in timings_online]

	return pct_complete_list,evolution.task_history,evolution


def plotInformation(fig,information,location,title,yaxis,maxy=None,dict_input=True,new_label=None,):

	ax = fig.add_subplot(location,adjustable='box')
	if dict_input == True: 
		task_information = {}
		num_tasks = len(information[-1])
		if len(information[0]) < num_tasks: 
			for i in np.arange(2): 
				for num in np.arange(num_tasks): 
					information[i][num] = 0
		#print information
		for taskid in np.arange(num_tasks): 
			task_information[taskid] = [x[taskid] for x in information]
		for taskid in np.arange(num_tasks):
			#print task_information[taskid]
			ax.plot(task_information[taskid],'-',label='Task '+str(taskid))
	else: 
		if new_label == None: 
			new_label = 'Estimate'
		#print information
		ax.plot(information,':',label=new_label)
		ax.plot(np.arange(len(information)),100*np.arange(len(information))/float(len(information)),'--',color='gray',label='Perfect')
	ax.set_title(title)
	ax.set_ylabel(yaxis)
	ax.legend(loc='best')
	if maxy != None:
		ax.set_ylim((0,maxy))
	return
	
def example(kinectDict,startsDict,txtnames,_plot=True): 
	import seaborn as sns 
	sns.set_style('whitegrid')
	sns.set_context("paper")
	sns.set_palette(sns.cubehelix_palette(3, start=2, rot=0.45, dark=0.2, light=.8, reverse=True))
	types,states,counts = [1,3,8,1,3,8],[2,2,2,4,4,4],[1,1,1,2,2,2]
	#types,states,counts = [1,7,9,1,7,9],[2,2,2,4,4,4],[1,1,1,1,1,1]
	#types,states,counts = [1,6,9,1,6,9],[1,1,1,2,2,2],[1,1,1,1,1,1]
	#types,states,counts = [1,1,1,1,1],[1,1,1,1,1],[1,8,8,8,8]
	pcts,tasks,evolution=testRun(kinectDict,startsDict,txtnames,types,states,counts,_plot=_plot)
	return pcts,tasks
	plt.show()

def getComparisonValues(kinectDict,startsDict,pair): 
	k1,k2 = kinectDict[pair[0]],kinectDict[pair[1]]			#kinectData objects
	s1,s2 = startsDict[pair[0]],startsDict[pair[1]]			#starts values







if __name__ == '__main__': setup()