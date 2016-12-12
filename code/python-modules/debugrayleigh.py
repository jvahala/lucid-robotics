'''
Code to debug the mixed rayleigh issues using personal_test dataset
'''

import numpy as np 
import personal_test as pt 
import matplotlib.pyplot as plt 
from copy import deepcopy
import process


'''basic setup 
txtnames,kinectDict,startsDict = pt.setup()
task_type = txtnames[0]
kinect = kinectDict[task_type]
starts = startsDict[task_type]
idnum = '1'
user = pt.initializeUser(idnum,kinect,starts)	

print 'Initializing'
init_task = process.Task(curr_data_obj, curr_extrema=[relevant_starts[0],relevant_starts[1]], k=3, basis_dim=2, first_task=True)
print 'init data_inds', init_task.data_inds
evolution = process.Process(init_task)

curr_task_data_inds = np.arange(relevant_starts[curr_task_num],relevant_starts[curr_task_num+1])
curr_task_data = curr_data_obj.data_array[curr_task_data_inds,:]

'''
def runUpdate(user, kinect, starts, num_updates):
	reload(process)
	user_copy = deepcopy(user)

	#initialize on the first task
	init_task = process.Task(kinect,[starts[0],starts[1]],first_task=True)
	evolution = process.Process(init_task)
	a = init_task.printTaskDef()
	a = [int(x) for x in a[1]]
	print a


	# use a random task instance order
	task_instances_to_use = np.random.permutation(range(1,len(starts)-1))[0:num_updates]
	
	for instance in task_instances_to_use: 
		# for each instance run through the necessary data inds
		curr_task_data_inds = np.arange(starts[instance],starts[instance+1])
		curr_task_data = kinect.data_array[curr_task_data_inds,:]

		# do the online process
		for i,data in enumerate(curr_task_data): 
			print 'Frame: '+str(i)
			new_features = pt.getNewFeatures(user_copy,data)
			pct_complete = evolution.onlineUpdate(new_features,user_copy)
			print 'Evolution pct complete: ', pct_complete
		print 'UPDATING_______\n\n'
		evolution.updateKnownTasks(user_copy,compute_type='average',proper_update=True)
		print 'DONE UPDATING'

	return 








