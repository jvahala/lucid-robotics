import process 
import main
from kinectData import kinectData
import numpy as np
import matplotlib.pyplot as plt 

def initialTask(idnum='4',initTaskNum=0):
	#collect all data from file to be fed into new data objects
	dummy_receiver,dummy_giver,starts,rtask,gtask = main.begin()

	#add necessary initialization to user object from dummy object
	userID = idnum
	user = kinectData(userID)
	user.names_list = dummy_receiver.names_list	

	#add initial task data
	init_task_inds = np.arange(starts[initTaskNum],starts[initTaskNum+1])
	user.data_array = dummy_receiver.data_array[init_task_inds,:]
	user.num_vectors = len(user.data_array)
	user.midpoint = dummy_receiver.midpoint
	user.similarity_method = dummy_receiver.similarity_method

	#user.feature_inds = dummy_receiver.feature_inds			#uncomment if explicit feature inds are wanted
	#get features for initial task  
	user.getFeatures(exp_weighting=True) 			#include some smoothing of data

	#initialize Task object
	print 'Initial Task Definition:'
	task = process.Task(user,[0,user.num_vectors])
	x = task.printTaskDef()

	return dummy_receiver,dummy_giver,starts,user,task

def initialize(): 
	#setup initialization variables
	userID = '4'
	initTaskNum = 0

	#setup other variables


	#initialize first task 
	dummy_receiver,dummy_giver,starts,user,task = initialTask(userID,initTaskNum)

	#define a process
	evolution = process.Process(task)

	return dummy_receiver,dummy_giver,starts,user,evolution

def play(dummy_receiver,dummy_giver,starts,user,evolution): 
	
	#perform online update process
	#for each new data element the current task 

	current_task_id = 0

	#plotting variables
	pct_values = {'est':[],0:[]}
	curr_labels = {0:[]}
	colors = 'kbgrmy'

	while current_task_id < 10: 
		current_task_id += 1
		curr_task_data_inds = np.arange(starts[current_task_id],starts[current_task_id+1])
		current_task_data = dummy_receiver.data_array[curr_task_data_inds,:]
		for data in current_task_data: 
			#add the data to the user object and get the new features
			user.data_array = np.vstack((user.data_array,data))
			user.num_vectors += 1
			#print user.data_array.shape
			user.getFeatures()
			#print user.all_features.shape
			new_features = user.all_features[-1,:]
			#use the Process object to iterate through all known tasks for 
			pct_complete = evolution.onlineUpdate(new_features,user)
			print '\n\n',evolution.task_pct_complete, 'Probable pct complete: ', pct_complete
			for x,t in evolution.known_tasks.iteritems(): 
				print 'Task ', x
				y = t.printTaskDef()
				print 'curr_state: ', t.curr_labels[-1]
				pct_values[x].append(evolution.task_pct_complete[x])
				curr_labels[x].append(t.curr_labels[-1])
			pct_values['est'].append(pct_complete)


		#once online process is complete, get cost of each known task
		evolution.updateKnownTasks(user)

		for x in pct_values: 
			plt.figure(current_task_id)
			if x == 'est': 
				linecolor = 'm'
				labeltext = 'Estimated Pct'
				curr_labels[x] = [-1]*len(pct_values[x])
			else: 
				linecolor = colors[x]
				labeltext = 'Option '+str(x)
			visualizePctIncrease(pct_values[x],curr_labels[x],linecolor,labeltext)
		plt.subplot(211)
		plt.legend(loc='lower right')
		plt.title('Task Number '+str(current_task_id))
		for x in evolution.known_tasks: 
			pct_values[x],curr_labels[x] = [],[]
		pct_values['est'],curr_labels['est'] = [],[]

	plt.show()





	return dummy_receiver,dummy_giver,starts,user,evolution


def visualizePctIncrease(pct_values,labels,linecolor='k',labeltext=''): 
	plt.subplot(211)
	if pct_values[0] == -1: 
		plt.plot(np.arange(len(pct_values)),pct_values,':',color=linecolor,label=labeltext)
	else:
		plt.plot(np.arange(len(pct_values)),pct_values,'-',color=linecolor,label=labeltext)
	plt.xlabel('')
	plt.ylim((0,100))
	plt.ylabel('Percent Complete')
	plt.subplot(212)
	if pct_values[0] != -1: 
		plt.plot(np.arange(len(pct_values)),labels,'-',color=linecolor,label=labeltext)
	plt.ylim((-1,4))
	plt.xlabel('Frame Number')
	plt.ylabel('State ID')





if __name__=='__main__': play()
