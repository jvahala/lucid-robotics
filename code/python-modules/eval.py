#import multi_test
from kinectData import kinectData
import numpy as np
import handover_tools as ht
import process
import matplotlib.pyplot as plt
from copy import deepcopy

def makeSingleTaskFile(task_number, base_path='/Users/vahala/Desktop/evaluation-data/',new_filename='test.txt'): 
	from genio import getlines,appendline
	new_filename_path = base_path+new_filename
	file_lineDicts = {}
	for i in np.arange(1,9): 
		file_append = 'p'+str(i)+'/txt-files/p'+str(i)+'-'+str(task_number)+'.txt'
		lines,c = getlines(base_path+file_append)
		print c
		if i == 1:
			with open(new_filename_path,'w') as f: 
				for i in np.arange(c): 
					new_line = lines[i]
					f.write(new_line)
		else: 
			with open(new_filename_path,'a') as f: 
				for i in np.arange(1,c):
					new_line = lines[i]
					f.write(new_line)

def makeAllSingleTaskFiles():
	for taskNum in np.arange(1,11): 
		filename='task-'+str(taskNum)+'.txt'
		makeSingleTaskFile(taskNum, new_filename=filename)

def defineUserWithData(task,user_id,base_path='/Users/vahala/Desktop/evaluation-data/',midpoint=None): 
	user = kinectData(str(user_id))
	user.addData(base_path+'task-'+str(task)+'.txt')
	user.similarity_method = 'exp'
	if midpoint != None: 
		user.midpoint = midpoint
	return user

def helpDetermineMidpoint(user,feature_name = 'HandLeft'): 
	''' plot the x,y,and z positions of the HandLeft feature to visually determine the handover point to minimize around'''
	from visualize import plotSingleFeature
	import seaborn as sns 
	sns.set_style('whitegrid')
	appenders = ['.X','.Y','.Z']
	xyz_inds = [user.names_list.index(feature_name+d) for d in appenders]

	#x axis 
	plt.figure(1)
	feature = user.data_array[:,xyz_inds[0]]
	plotSingleFeature(feature, 'HandLeft.X', color_choice = 'r')
	#y axis
	plt.figure(2)
	feature = user.data_array[:,xyz_inds[1]]
	plotSingleFeature(feature, 'HandLeft.Y', color_choice='g')
	#z axis
	plt.figure(3)
	feature = user.data_array[:,xyz_inds[2]]
	plotSingleFeature(feature, 'HandLeft.Z', color_choice='b')

def defineInitStarts(users): 
	import handover_tools as ht
	starts = {}
	for i in np.arange(1,9): 
		starts[i] = ht.handoverIDSingle(users[i],i=i)
		print 'User '+str(i)+' starts: ', starts[i], len(starts[i])
	return starts

def updateInitStarts(starts,taskID): 
	nstarts = {}
	if taskID == 1: 
		nstarts[1] = starts[1][1:]+[108,278]
		nstarts[2] = starts[2]+[57,84,133,164,228,258,287,324,355]
		nstarts[3] = starts[3]+[71]
		nstarts[4] = starts[4]+[48,112,178]
		nstarts[5] = starts[5]+[44,72,109,152,192,272,304,378,421]
		nstarts[6] = starts[6]+[52]
		nstarts[7] = starts[7]+[73]
		nstarts[8] = starts[8]+[71]
	elif taskID == 2: 
		nstarts[1] = starts[1][1:]+[97]
		nstarts[2] = starts[2]+[55,86,123,161,231,271,306,345,385]
		nstarts[3] = starts[3]+[37,96,264]
		nstarts[4] = starts[4]+[52,79,109,138,167,197,230,262,292,350]
		nstarts[5] = starts[5]+[54]
		nstarts[6] = starts[6]
		nstarts[7] = starts[7]+[61]
		nstarts[8] = starts[8]+[65,159,315,373]
	elif taskID == 3: 
		nstarts[1] = starts[1][1:]+[91,154,508,637,706,767]
		nstarts[2] = starts[2]+[62,93,122,164,196,231,267,314,356,392]
		nstarts[3] = starts[3]+[57]
		nstarts[4] = starts[4]+[44]
		nstarts[5] = starts[5]+[48,113,148,190,275,382]
		nstarts[6] = starts[6]+[104,145,185,229,275,319,365,411,454,495]
		nstarts[7] = starts[7]+[62]
		nstarts[8] = starts[8]+[39,100,158,220,279,338]
	elif taskID == 4: 
		nstarts[1] = starts[1][0:-2]+[starts[1][-1]]+[60,832]
		nstarts[2] = starts[2]+[52]
		nstarts[3] = starts[3]+[37]
		nstarts[4] = starts[4]+[49]
		nstarts[5] = starts[5]+[49]
		nstarts[6] = starts[6]+[112]
		nstarts[7] = starts[7]+[48]
		nstarts[8] = starts[8]+[47]
	elif taskID == 5: 
		nstarts[1] = starts[1][0:8]+starts[1][9:]
		nstarts[2] = starts[2]+[48]
		nstarts[3] = starts[3]+[39]
		nstarts[4] = starts[4]+[52]
		nstarts[5] = starts[5]+[57]
		nstarts[6] = starts[6][1:]+[90]
		nstarts[7] = starts[7]+[49]
		nstarts[8] = starts[8]+[49]
	elif taskID == 6: 
		nstarts[1] = starts[1]+[108]
		nstarts[2] = starts[2][0:5]+[41,327,379,429,477,525]
		nstarts[3] = starts[3]+[54]
		nstarts[4] = starts[4]+[60]
		nstarts[5] = starts[5]+[58]
		nstarts[6] = [62,104,148,199,250,301,350,392,438,482,526]
		nstarts[7] = starts[7]+[51]
		nstarts[8] = starts[8]+[59]
	elif taskID == 7: 
		nstarts[1] = [76,206,392,519,660,795,917,1021,1145,1261,1377]
		nstarts[2] = starts[2]+[58]
		nstarts[3] = [starts[3][0]]+starts[3][2:4]+[32,332,410,484,577,638,725,823]
		nstarts[4] = starts[4][1:4]+starts[4][5:9]+[starts[4][10]]+starts[4][12:14]+[56]
		nstarts[5] = starts[5]+[45]
		nstarts[6] = starts[6]+[52]
		nstarts[7] = starts[7][0:6]+[52,631,711,792,881]
		nstarts[8] = starts[8]+[54]
	elif taskID == 8: 
		nstarts[1] = [96,225,374,522,658,816,961,1109,1248,1396,1558]
		nstarts[2] = [45,138,245,358,474,604,729,855,979,1114,1230]
		nstarts[3] = starts[3]+[41]
		nstarts[4] = [starts[4][0]]+starts[4][2:]+[58]
		nstarts[5] = starts[5]+[50]
		nstarts[6] = starts[6]+[50]
		nstarts[7] = starts[7][0:-1]+[64]
		nstarts[8] = starts[8]+[50]
	elif taskID == 9: 
		nstarts[1] = [39,165,266,407,527,652,773,891,996,1126,1258]
		nstarts[2] = starts[2][0:4]+[starts[2][5]]+[starts[2][7]]+[starts[2][9]]+[45,924,1033,1146]
		nstarts[3] = starts[3][0:3]+starts[3][4:10]+[52,390]
		nstarts[4] = starts[4][0:2]+starts[4][3:5]+[starts[4][6]]+starts[4][8:12]+[starts[4][13]]+[44]
		nstarts[5] = [starts[5][1]]+starts[5][3:8]+starts[5][9:12]+[starts[5][13]]+[44]
		nstarts[6] = starts[6][0:-1]+[61]
		nstarts[7] = starts[7][0:3]+[starts[7][4]]+starts[7][6:10]+[starts[7][11]]+[58,913]
		nstarts[8] = [55,133,217,304,396,479,568,645,733,817,912]
	elif taskID == 10: 
		nstarts[1] = [96,235,358,483,645,780,903,1058,1193,1325,1467]
		nstarts[2] = [64,161,276,378,508,610,725,841,960,1092,1256]
		nstarts[3] = starts[3]+[41]
		nstarts[4] = starts[4]+[74]
		nstarts[5] = [starts[5][0]]+starts[5][2:8]+starts[5][9:12]+[50]
		nstarts[6] = starts[6]+[43]
		nstarts[7] = starts[7][0:-1]+[77]
		nstarts[8] = starts[8]+[58]

	if len(nstarts)==0:
		nstarts = starts 
	else: 
		for k,v in nstarts.iteritems():
			v.sort()

	return nstarts 

def defineTaskDicts():
	'''this should be run as an initialization procedure to get all data into dummy locations
	taskDictUsers - {taskID: {userID: kinectData object}} for all tasks and users 
	taskDictStarts - {taskID: {userID: [starts list]}} for all tasks and users, starts list has 11 points in it representing 10 iterations of the task
	'''
	midpoint = [-0.45,0.13, 2.05]		#found by looking at .X,.Y,.Z HandLeft position - same for all users
	taskDictUsers = {}
	taskDictStarts = {}
	for i in np.arange(1,11):
		starts = {}
		users = {}

		for k in np.arange(1,9):
			users[k] = defineUserWithData(task=i,user_id=k,midpoint=midpoint)
			users[k].getFeatures()

		starts = defineInitStarts(users)
		print 'TASK', i
		plt.show()
		starts = updateInitStarts(starts,taskID=i)

		taskDictUsers[i] = users 
		taskDictStarts[i] = starts
	print taskDictUsers
	print taskDictStarts 

	return taskDictUsers,taskDictStarts

def getStartIndsFromIterID(starts,iter_id): 
	return [starts[iter_id],starts[iter_id+1]]

def initializeUser(userID,init_object,init_starts):
	#create an object
	user = kinectData(str(userID))

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

def dealWithEvolution(taskDictUsers,taskDictStarts,userID,task_plan,evolution=None,user=None): 
	'''
	Purpose: 
	intialize a process.Process object with the input task_plan defined as a dictionary of task ID keys with iteration numbers as values

	Inputs: 
	taskDictUsers - returned from defineTaskDicts()
	taskDictStarts - returned from defineTaskDicts()
	userID - int user number to initialize on 
	task_plan - dict {taskID:[iteration numbers], taskID2:[iteration numbers 2], ...}, taskId in {1,...,10}

	Outputs: 
	evolution - process.Process() object 
	'''
	if evolution != None: 
		initialized = True
	else: 
		initialized = False
		pct_complete = 0 

	#print 'Task plan: ', task_plan

	for taskID,iters_list in task_plan.iteritems(): 
		#print 'Initializing Task '+str(taskID)+' with '+str(len(iters_list))+' iterations'
		curr_data_obj = taskDictUsers[taskID][userID]		#grab the user id kinectData object of the current task 
		curr_starts = taskDictStarts[taskID][userID]		#grab the starts for this users in the current task as well
		
		for iter_id in iters_list: 		#run through the proper iterations {0,1,...,9}
			
			#get the current begin and end frame numbers
			[begin,end] = getStartIndsFromIterID(curr_starts,iter_id)

			#initialize if first task seen
			if not initialized: 
				init_task = process.Task(curr_data_obj, curr_extrema=[begin,end], k=3, basis_dim=2, first_task=True)
				#a = init_task.printTaskDef()
				#a = [int(x) for x in a[1]]
				#print a
				evolution = process.Process(init_task)
				user = initializeUser(userID,curr_data_obj,[begin,end])
				initialized = True
			else: 
				#print 'ERRor stuff: ', userID, curr_data_obj.ID, curr_data_obj.data_array.shape, begin, end
				curr_task_data = curr_data_obj.data_array[np.arange(begin,end),:]
				for data in curr_task_data: 
					new_features = getNewFeatures(user,data)
					pct_complete = evolution.onlineUpdate(new_features,user)
				evolution.updateKnownTasks(user,compute_type='average')

	return user,evolution,pct_complete

def runSingleUser(taskDictUsers,taskDictStarts,userID,max_unique=10,max_iters=10,average_over=50,seedint=21): 
	np.random.seed(seedint)
	num_to_average = average_over

	pcts_known = np.zeros((max_unique,max_iters))
	pcts_unknown = np.zeros((max_unique,max_iters))
	update_counts_known = np.zeros((max_unique,max_iters))
	update_counts_unknown = np.zeros((max_unique,max_iters))
	std_known = np.zeros((max_unique,max_iters))
	std_unknown = np.zeros((max_unique,max_iters))
	meds_known = np.zeros((max_unique,max_iters))
	meds_unknown = np.zeros((max_unique,max_iters))

	#for 1 to 9 unique tasks to go through
	all_pcts = {}
	all_ukpcts = {}
	for i in np.arange(1,max_unique):
		all_pcts[i] = {}
		all_ukpcts[i] = {}
		#for 1 to 9 iterations of a task to train on 
		for j in np.arange(1,max_iters): 
			all_pcts[i][j] = []
			all_ukpcts[i][j] = []
			print 'pct k: \n', pcts_known
			print 'cnt k: \n', update_counts_known
			print 'pct un: \n', pcts_unknown
			print 'cnt un: \n', update_counts_unknown
			print 'med k: \n', meds_known
			print 'med un: \n', meds_unknown
			print 'std k: \n', std_known
			print 'std un: \n', std_unknown
			print '\n\nBegin averaging ----------------------------', i,j
			#do this for n tries, and average the final results 
			n = 0
			pk,puk,ck,cuk = 0,0,0,0
			all_pk,all_puk = [],[]
			while n < num_to_average: 
				n+= 1
				print '\nn = '+str(n)
				task_plan = {}
				#create a task path permutation 
				full_task_path = np.random.permutation(10)+1
				#if i>7:
					#break
				#full_task_path = np.random.permutation([1,2,3,4,5,6,7,8])
				
				#known task_path 
				#print 'full task path', full_task_path, i, full_task_path[0:i]
				task_path = full_task_path[0:i]

				#choose j iterations to go through for each task in the task path
				for taskID in task_path: 

					task_plan[taskID] = list(np.random.permutation(10)[0:j])

				#initialize evolution 
				user_known, evolution_known, pct_complete = dealWithEvolution(taskDictUsers,taskDictStarts,userID,task_plan,evolution=None,user=None)

				#make a deep copy of the user and evolution state for use with the unknown test
				user_unknown = deepcopy(user_known)
				evolution_unknown = deepcopy(evolution_known)
				task_count = deepcopy(evolution_known.known_task_count)

				#select new known task 
				known_task_choice = task_path[np.random.randint(i)]
				known_iter_choice = np.random.randint(10)

				#process known task 
				#print 'Testing ~known~ Task'
				known_task_plan = {known_task_choice:[known_iter_choice]}
				user_known, evolution_known, pct_complete = dealWithEvolution(taskDictUsers,taskDictStarts,userID,known_task_plan,evolution=evolution_known,user=user_known)
				print 'Known Task: '+str(known_task_choice)+'\t PCT known: ', pct_complete
				all_pk.append(min(pct_complete,100.0))
				all_pcts[i][j].append(min(pct_complete,100.0))
				pk += min(pct_complete,100.0)
				ck += (evolution_known.known_task_count - task_count)

				#select unknown task 
				unknown_task_choice = full_task_path[-1]
				unknown_iter_choice = known_iter_choice 		#may as whell not compute this int twice when it doesn't matter

				#process unknown task 
				#print 'Testing !UNKNOWN! Task'
				unknown_task_plan = {unknown_task_choice:[unknown_iter_choice]}
				user_unknown, evolution_unknown, pct_complete = dealWithEvolution(taskDictUsers,taskDictStarts,userID,unknown_task_plan,evolution=evolution_unknown,user=user_unknown)
				print 'PCT unknown: ', pct_complete
				all_puk.append(min(pct_complete,100.0))
				all_ukpcts[i][j].append(min(pct_complete,100.0))
				puk += min(pct_complete,100.0)
				cuk += (evolution_unknown.known_task_count - task_count)
				user_known, user_unknown, evolution_known, evolution_unknown = None, None, None, None
			
			#collect avg, std, median values for known and unknown tasks
			pcts_known[i,j] = pk/float(n)
			std_known[i,j] = np.std(all_pk)
			meds_known[i,j] = np.median(all_pk)

			pcts_unknown[i,j] = puk/float(n)
			std_unknown[i,j] = np.std(all_puk)
			meds_unknown[i,j] = np.median(all_puk)

			update_counts_known[i,j] = 100*ck/float(n)
			update_counts_unknown[i,j] = 100*cuk/float(n)

			#print all_pcts[i][j]

	return pcts_known, pcts_unknown, update_counts_known, update_counts_unknown, std_known,std_unknown,meds_known,meds_unknown, all_pcts, all_ukpcts

def testAgainstUsers(taskDictUsers,taskDictStarts,iterations=10,seedint=1):
	
	np.random.seed(seedint)
	out_pcts = np.zeros((iterations,10))
	out_add_rate = np.zeros((iterations,10))
	#for each iteration, choose a training user 
	for i in np.arange(iterations): 
		if i == 0: 
			user_set = np.random.permutation(8)+1
		else: 
			user_set = np.vstack((user_set,np.random.permutation(8)+1))

	#main loop, for each iteration test each task with the primary user then test with all the rest of the users
	for i,curr_user_set in enumerate(user_set): 
		print 'Current User Set: ', i+1, 'of', len(user_set)

		#for each task number 1 through 10, do this 
		for taskID in np.arange(1,11):
			print 'Current progress: ', str(i)+'-'+str(taskID), 'user'+str(curr_user_set[0])
			#create the initialized user and evolution
			task_plan = {taskID:list(np.random.permutation(10))}
			user, evolution, pct = dealWithEvolution(taskDictUsers, taskDictStarts, curr_user_set[0], task_plan)

			start_task_count = deepcopy(evolution.known_task_count)
			task_added_count = 0
			#for each other user
			for other_user in curr_user_set[1:]: 
				print 'user'+str(other_user)
				#create copy of the initialized state
				user_copy, evolution_copy = deepcopy(user), deepcopy(evolution)
				#update
				task_plan = {taskID:[np.random.randint(10)]}
				user_copy, evolution_copy, new_pct = dealWithEvolution(taskDictUsers,taskDictStarts,other_user,task_plan,evolution_copy,user_copy)
				task_added_count += evolution_copy.known_task_count - start_task_count
				user_copy, evolution_copy = None, None
				out_pcts[i,taskID-1] = deepcopy(new_pct) 
			out_add_rate[i,taskID-1] = 100*task_added_count/9.

			user, evolution = None, None

	return out_pcts, out_add_rate 
		



def plotAnArray(array_obj): 
	import matplotlib.pyplot as plt
	import seaborn as sns
	sns.set_style('whitegrid')
	sns.set_context("paper")
	sns.set_palette(sns.cubehelix_palette(5, start=2, rot=0.45, dark=0.2, light=.8, reverse=True))
	for i,row in enumerate(array_obj[1:,1:]): 
		plt.plot(np.arange(array_obj.shape[1]-1),row,marker='o',label='q = '+str(i))

def getWhatIWant(taskDictUsers,taskDictStarts): 
	pct_known,pct_unknown,c_known,c_unknown,std_known,std_unknown,meds_known,meds_unknown = [],[],[],[],[],[],[],[]
	all_pcts,all_ukpcts = [],[]
	for i in np.arange(1,9): 
		print '\n\nuser ', i
		np_k, np_uk, nc_k, nc_uk, nv_k, nv_uk, nm_k, nm_uk, a_pct, a_ukpct = runSingleUser(taskDictUsers,taskDictStarts,userID=i,max_unique=5,max_iters=5,average_over=30,seedint=1)
		pct_known.append(np_k)
		pct_unknown.append(np_uk)
		c_known.append(nc_k)
		c_unknown.append(nc_uk)
		std_known.append(nv_k)
		std_unknown.append(nv_uk)
		meds_known.append(nm_k)
		meds_unknown.append(nm_uk)
		all_pcts.append(a_pct)
		all_ukpcts.append(a_ukpct)

	return pct_known,pct_unknown,c_known,c_unknown,std_known,std_unknown,meds_known,meds_unknown,all_pcts,all_ukpcts

def combineUserPcts(all_pcts): 
	'''
	Purpose: 
	takes the all_pcts output of getWhatIWant(), which looks like [all_pcts(user1), all_pcts(user2), ...] and combines them into a single all_pcts not separated by user. Thus, if an individual user had n=30 examples of pcts and there were 4 users, then the new all_pcts for each datapoint would have 120 examples. 

	Inputs: 
	all_pcts - k-user length list of all_pcts outputs of runSingleUser()
	each elements looks like dict[~userid(0 to 7)~] [~uniqueTasks(1 to 4)~] [~iterationsTrained(1 to 4)] and this last dictionary contains a list of percents

	Outputs: 
	combined_pcts - '''
	combined = {}	#indexed by q
	for userid,userdata in enumerate(all_pcts): 
		#want to take the values for each user and place them together, will be plotting the uniqueTasks as different sets
		for numuniquetasks, uniquetaskdata in userdata.iteritems(): 
			#if there was only 1 unique task, then I'd want to grab that data as four points
			if userid == 0: 
				combined[numuniquetasks] = {} 
			for numiterations, iterationsdata in uniquetaskdata.iteritems(): 
				if userid == 0: 
					combined[numuniquetasks][numiterations] = iterationsdata	#list
				else: 
					combined[numuniquetasks][numiterations] += iterationsdata
	return combined

def plotCombined(combined,plot_inds=None,linestyle=None,figure=None,success_threshold=80,success_shift_x=1,success_shift_y=-1.33): 
	'''
	Purpose: 
	Plot the 'combined' object returned by combineUserPcts(). 

	Inputs: 
	combined - dictionary return value of combineUserPcts()
	plot_inds - only plots inds found in this list, defaults to [1,2,...len(combined[1])]
	linestyle - linestyle, defaults to '-'
	figure - figure number, defaults to 1
	success_threshold - horizontal line placed at this value, defaults to 80
	success_shift_x - 0-100 input, represents percent-total-visible-x-axis location of 'Success Threshold' text label for the threshold line, defaults to 1
	success_shift_y - y-axis value at which to place the 'Success Threshold' text
	'''
	
	if plot_inds == None: 
		plot_inds = range(1,len(combined[1])+1)
	if linestyle == None: 
		linestyle = '-'
	if figure == None: 
		figure = 1

	import matplotlib.pyplot as plt
	import matplotlib.lines as mlines
	import seaborn as sns; sns.set(color_codes=True)
	sns.set_style('whitegrid')
	sns.set_context('paper')
	algae = '#21c36f'
	twilight = '#0a437a'
	sunflower = '#ffc512'

	lightblue = '#95d0fc'
	teal = '#029386'
	aqua = '#13eac9'
	darkblue = '#00035b'
	colors = ['k',darkblue,teal,aqua,lightblue]
	mline1 = mlines.Line2D([], [], color=colors[1], label='q = 1')
	mline2 = mlines.Line2D([], [], color=colors[2], label='q = 2')
	mline3 = mlines.Line2D([], [], color=colors[3], label='q = 3')
	mline4 = mlines.Line2D([], [], color=colors[4], label='q = 4')
	all_mlines=[mline1,mline2,mline3,mline4]

	fig = plt.figure(figure)
	ax = fig.add_subplot(111)
	data = np.zeros((len(combined[1][1]),len(combined[1])))
	for q,v in combined.iteritems(): 
		if q not in plot_inds: 
			continue
		#q is number of unique tasks in the test, v contains a dictionary indexed by the number of iterations trained. 
		for k,r in v.iteritems(): 
			data[:,k-1] = r 
		sns.tsplot(data=data,time=range(1,k+1),estimator=np.median,err_style='ci_band',ci=[95],linestyle=linestyle,marker='o',color=colors[q])
	
	x_plot_min,x_plot_max = 0.2,k+0.4
	plt.plot(np.array([0,k+2]),np.ones((2))*success_threshold,'k--')
	plt.text(x_plot_min+success_shift_x*(x_plot_max-x_plot_min)/100.,success_threshold+success_shift_y,'Success Threshold',fontsize='x-small',fontweight='demibold')

		#mean = np.mean(data,axis=0) 
		#std = np.std(data,axis=0)
		#ax.errorbar(range(1,k+1),mean,yerr=std,fmt='none')


	plt.xlabel('Number of Iterations Trained (n)')
	plt.ylabel('Percent Complete Estimate at New Task End (%)')
	include_handles = [x for i,x in enumerate(all_mlines) if i+1 in plot_inds]
	plt.legend(handles=include_handles)
	plt.ylim((60,102))
	plt.xlim((0.2,4.4))
	ax.set_xticks(range(1,k+1))
	return data



	# for k,v in qbases_k.iteritems(): 
	# 	sns.tsplot(data=v[:,1:],time=range(1,len(v)),err_style='ci_bars',marker='o',color=colors[k])
	# for k,v in qbases_uk.iteritems(): 
	# 	sns.tsplot(data=v[:,1:],time=range(1,len(v)),linestyle='--',err_style='ci_bars',marker='o',color=colors[k])
	# plt.text(0.72,78,'Known',size=10, style='italic',rotation='vertical')		#0.72,78 - qpcts, 12 cnts
	# plt.text(0.72,70,'Unknown',size=10, style='italic', rotation='vertical')	#0.72,70 - qpcts, 66 cnts
	# plt.xlabel('Number of Iterations Trained (n)')
	# plt.ylabel('Final Percent Complete for Test Iteration (%)')
	# #plt.ylabel('Probability of Defining New Task Model')
	# #plt.legend(handles=[mline1,mline2,mline3,mline4], loc='best')
	# plt.legend(handles=[mline1], loc='best')
	# plt.ylim((0,100))		#60,95 for qpcts, remove for qcnts
	# plt.xlim((0.6,4.4))		#0.5,6.5 for qpcts
	# ax.set_xticks(range(1,len(v)))
	# plt.show()



def turnIntoQbase(listofpcts):
	qpcts= {}
	for i,pcts in enumerate(listofpcts): 
		#print 'list of ', pcts
		if i == 0: 
			for r,row in enumerate(pcts):
				if r >0: 
					#print 'pcts r ', pcts[r]
					qpcts[r] = pcts[r]
			print 'qpcts init: \n', qpcts
		else: 
			for r,row in enumerate(pcts):
				if r>0: 
					qpcts[r] = np.vstack((qpcts[r],row))
	return qpcts 

def plotInterTask(pcts,addrates):
	import matplotlib.pyplot as plt
	import matplotlib.lines as mlines
	import seaborn as sns; sns.set(color_codes=True)
	sns.set_style('whitegrid')
	sns.set_context('paper')
	lightblue = '#95d0fc'
	teal = '#029386'
	aqua = '#13eac9'
	darkblue = '#00035b'
	colors = ['k',darkblue,teal,aqua,lightblue]
	mline1 = mlines.Line2D([], [], color=colors[1],marker='o', label='Percent at New Task End')
	mline3 = mlines.Line2D([], [], color=colors[3],marker='s', label='New Task Model Add Rate')

	fig, ax = plt.subplots(1, 1)
	sns.tsplot(data=pcts, time=np.arange(1,11), interpolate=False, err_style='ci_bars',marker='o',color=colors[1],ax=ax)
	sns.tsplot(data=addrates, time=np.arange(1,11), interpolate=False, err_style='ci_bars',marker='s',color=colors[3],ax=ax)
	ax.set_xticks(np.arange(1,11))
	plt.xlabel('Task ID')
	plt.ylabel('Percent or Rate')
	plt.legend(handles=[mline1,mline3],loc='lower right', frameon=True)
	plt.show()


def plotQbase(qbases_k, qbases_uk):
	#http://stackoverflow.com/questions/38385099/adding-simple-error-bars-to-seaborn-factorplot
	import matplotlib.pyplot as plt
	import matplotlib.lines as mlines
	import seaborn as sns; sns.set(color_codes=True)
	sns.set_style('whitegrid')
	sns.set_context('paper')
	algae = '#21c36f'
	twilight = '#0a437a'
	sunflower = '#ffc512'

	lightblue = '#95d0fc'
	teal = '#029386'
	aqua = '#13eac9'
	darkblue = '#00035b'
	colors = ['k',darkblue,teal,aqua,lightblue]
	mline1 = mlines.Line2D([], [], color=colors[1], label='q = 1')
	mline2 = mlines.Line2D([], [], color=colors[2], label='q = 2')
	mline3 = mlines.Line2D([], [], color=colors[3], label='q = 3')
	mline4 = mlines.Line2D([], [], color=colors[4], label='q = 4')

	fig, ax = plt.subplots(1,1)
	for k,v in qbases_k.iteritems(): 
		sns.tsplot(data=v[:,1:],time=range(1,len(v)),err_style='ci_bars',marker='o',color=colors[k])
	for k,v in qbases_uk.iteritems(): 
		sns.tsplot(data=v[:,1:],time=range(1,len(v)),linestyle='--',err_style='ci_bars',marker='o',color=colors[k])
	plt.text(0.72,78,'Known',size=10, style='italic',rotation='vertical')		#0.72,78 - qpcts, 12 cnts
	plt.text(0.72,70,'Unknown',size=10, style='italic', rotation='vertical')	#0.72,70 - qpcts, 66 cnts
	plt.xlabel('Number of Iterations Trained (n)')
	plt.ylabel('Final Percent Complete for Test Iteration (%)')
	#plt.ylabel('Probability of Defining New Task Model')
	#plt.legend(handles=[mline1,mline2,mline3,mline4], loc='best')
	plt.legend(handles=[mline1], loc='best')
	plt.ylim((0,100))		#60,95 for qpcts, remove for qcnts
	plt.xlim((0.6,4.4))		#0.5,6.5 for qpcts
	ax.set_xticks(range(1,len(v)))
	plt.show()

#want to turn rows into specific arrays 

# 1. run:	taskDictUsers,taskDictStarts = defineTaskDicts(), alternatively can get starts from all_starts.txt located with the eval data
# 2. define: task_plan = {task1:[iter1,iter2,...],task2:[iter1,iter2], ...}
# 2. run:	evolution = initializeEvolution(taskDictUsers,taskDictStarts,userID=xxx,task_plan)

'''Process for running an evaluation
1. define starts and task dicts, midpoint has been determined by sight and with the helper functions in this module
taskDictUsers,taskDictStarts = defineTaskDicts()
2a. If you want to do a single user evaluation, run: 
userID = #choose a number 
pcts_known,pcts_unknown,update_counts_known,update_counts_unknown = runSingleUser(taskDictUsers,taskDictStarts,userID,max_unique=10,max_iters=10,average_over=50,seedint=21)	#set these defaults appropriately
### 2a-i if you want to run all single users, (note you will need to change defaults for runSingleUser within this function)
pct_known,pct_unknown,c_known,c_unknown=getWhatIWant(taskDictUsers,taskDictStarts)
2b. If you want to compare across users, run: 
out_pcts, out_add_rate = testAgainstUsers(taskDictUsers,taskDictStarts,iterations=10,seedint=1)
3a. plot 2a-i: 
3b. plot 2b: plotInterTask(out_pcts, out_add_rate)

'''


# mdpt = np.array([-0.45,0.13, 2.05])

