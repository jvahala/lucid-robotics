import numpy as np
import matplotlib.pyplot as plt
import utils

def handoverIDSingle(user,i=None): 
	''' For getting the handover statres for a single user scenario '''
	joint_left = 'HandLeft'
	joint_right = 'HandRight'

	left = getIndivJoint(user,joint_left)
	right = getIndivJoint(user,joint_right)

	lmid = getJointDifference(left,user.midpoint)
	rmid = getJointDifference(right,user.midpoint)

	N = 10 
	lmid = utils.runningAvg(lmid,N)
	rmid = utils.runningAvg(rmid,N)

	lvel,lacc = handoverSpeed(lmid,N)
	rvel,racc = handoverSpeed(rmid,N)

	#determine probable areas for handover splits using thresholds and a version of gradient descent to find handtohand distance minimums
	min_dist = 0.4	#meters
	min_vel = 0.25	#meters/sec
	lsplit = np.logical_and(lmid[0:-1] < min_dist, np.abs(lvel)<min_vel)
	rsplit = np.logical_and(rmid[0:-1] < min_dist, np.abs(rvel)<min_vel)
	
	l_inds = [ind for ind,x in enumerate(lsplit) if x==True]
	l_inds = groupLikeIndex(l_inds,len(lacc))
	l_inds = [minDistIndex(x,lmid) for x in l_inds]

	r_inds = [ind for ind,x in enumerate(rsplit) if x==True]
	r_inds = groupLikeIndex(r_inds,len(racc))
	r_inds = [minDistIndex(x,rmid) for x in r_inds]

	temp_starts = np.sort(l_inds + r_inds)
	handover_starts = []
	value = 10			#was 20 for personal starts
	for start in temp_starts: 
		if np.abs(start-value)>30: 		#was 70 for personal starts
			value = start
			handover_starts.append(start)

	print 'final starts: ', handover_starts

	l_inds,r_inds = handover_starts,handover_starts
	# if i != None: 
	# 	plt.figure(i)
	# else: 
	# 	plt.figure(1)
	# l_inds = [x-1 for x in l_inds]
	# plotHandoverInfo(lmid,lvel,lacc,l_inds,color_str='r',label='Left Hand')
	# plotHandoverInfo(rmid,rvel,racc,r_inds,color_str='b',label='Right Hand')
	# plt.legend()

	return handover_starts

def handoverID(user1, user2):
	'''
	Purpose: 
	Identifies the frame numbers at which handovers occur between two human users. Handovers are defined as starting at the zero velocity point when two users hands are closest to each other. 

	Inputs: 
	user1 - kinectData class of object for user1 
	user2 - kinectData class of object for user2

	Outputs:
	handover_starts - list of frame numbers associated with handover starts 

	'''

	#get features associated with hand positions 
	joint_left = 'HandLeft'
	left1 = getIndivJoint(user1,joint_left)
	left2 = getIndivJoint(user2,joint_left)

	joint_right = 'HandRight'
	right1 = getIndivJoint(user1,joint_right)
	right2 = getIndivJoint(user2,joint_right)

	#create new variables associated with all difference potential handovers (user 1 left hand to user 2 right hand, etc)
	ll = getJointDifference(left1,left2)
	lr = getJointDifference(left1,right2)
	rl = getJointDifference(right1,left2)
	rr = getJointDifference(right1,right2)
	N = 10
	ll = utils.runningAvg(ll,N)
	lr = utils.runningAvg(lr,N)
	rl = utils.runningAvg(rl,N)
	rr = utils.runningAvg(rr,N)

	#compute velocites and accelerations

	llvel, llacc = handoverSpeed(ll,N)
	lrvel, lracc = handoverSpeed(lr,N)
	rlvel, rlacc = handoverSpeed(rl,N)
	rrvel, rracc = handoverSpeed(rr,N)

	#determine probable areas for handover splits using thresholds and a version of gradient descent to find handtohand distance minimums
	min_dist = 0.4	#meters
	min_vel = 0.5	#meters/sec
	llsplit = np.logical_and(ll[0:-1] < min_dist, np.abs(llvel)<min_vel)
	lrsplit = np.logical_and(lr[0:-1] < min_dist, np.abs(lrvel)<min_vel)
	rlsplit = np.logical_and(rl[0:-1] < min_dist, np.abs(rlvel)<min_vel)
	rrsplit = np.logical_and(rr[0:-1] < min_dist, np.abs(rrvel)<min_vel)
	
	ll_inds = [ind for ind,x in enumerate(llsplit) if x==True]
	ll_inds = groupLikeIndex(ll_inds,len(llacc))
	ll_inds = [minDistIndex(x,ll) for x in ll_inds]

	lr_inds = [ind for ind,x in enumerate(lrsplit) if x==True]
	lr_inds = groupLikeIndex(lr_inds,len(lracc))
	lr_inds = [minDistIndex(x,lr) for x in lr_inds]

	rl_inds = [ind for ind,x in enumerate(rlsplit) if x==True]
	rl_inds = groupLikeIndex(rl_inds,len(rlacc))
	rl_inds = [minDistIndex(x,rl) for x in rl_inds]

	rr_inds = [ind for ind,x in enumerate(rrsplit) if x==True]
	rr_inds = groupLikeIndex(rr_inds,len(rracc))
	rr_inds = [minDistIndex(x,rr) for x in rr_inds]

	handover_starts = ll_inds+lr_inds+rl_inds+rr_inds
	
	#print 'starts', handover_starts
	'''
	#plot results
	plt.figure(1)
	plotHandoverInfo(ll,llvel,llacc,ll_inds,color_str='r')
	plotHandoverInfo(lr,lrvel,lracc,lr_inds,color_str='b')
	plotHandoverInfo(rl,rlvel,rlacc,rl_inds,color_str='g')
	plotHandoverInfo(rr,rrvel,rracc,rr_inds,color_str='y')
	plt.legend(['left left','left right', 'right left', 'right right'])
	plt.show()
	'''
	return handover_starts

def groupLikeIndex(ind_list, max_ind):
	'''
	Purpose: 
	Takes input of a list of indices and turns them into a sparse set of indices where no 2 are very close

	Inputs: 
	ind_list - list of indices 
	max_ind - maximum int value allowed for any single index
	
	Outputs: 
	grouped - sparse set of indices
	'''
	temp = ind_list
	grouped = [x for ind,x in enumerate(ind_list) if (np.abs(x-temp[ind-1])>20 or ind==0)]
	if len(grouped)>0 and grouped[-1] >= max_ind:
		return grouped[0:-1]
	return grouped

def plotHandoverInfo(handtohand,handvel,handacc,hand_inds,color_str,label):
	'''
	Purpose: 
	plots in a 3 by 1 subplot the distance, velocity, acceleration, and key indices for the handtohand difference between users

	Inputs: 
	handtohand -  ndarray, ||(hand1-hand2)||_2 
	handvel - ndarray, velocity of handtohand 
	handacc - ndarray, acceleration of handtohand
	hand_inds - indices defined to be the potential handover start locations
	color_str - color to make the plot markings, ie 'red' or 'k' 
	
	Outputs: 
	plot, must call legend outside of plot. 
	'''

	spf = 1. 		#seconds per frame
	plt.subplot(3,1,1)
	plt.plot(spf*np.arange(len(handvel)),handtohand[0:-1],color=color_str,label=label)
	plt.plot(spf*np.array(hand_inds),handtohand[hand_inds],linestyle='None',marker='x',color=color_str,label='_nolegend_')
	plt.ylabel('distance (m)')
	plt.title('giver-to-receiver hand-to-hand stats')
	plt.subplot(3,1,2)
	plt.plot(spf*np.arange(len(handvel)),handvel,color=color_str,label=label)
	plt.plot(spf*np.arange(len(handvel)),np.zeros((len(handvel),1)),color='k',label='_nolegend_')
	plt.plot(spf*np.array(hand_inds),handvel[hand_inds],linestyle='None',marker='x',color=color_str,label='_nolegend_')
	plt.ylabel('velocity (m/s)')
	plt.subplot(3,1,3)
	plt.plot(spf*np.arange(len(handacc)),handacc,color=color_str,label=label)
	plt.plot(spf*np.arange(len(handacc)),np.zeros((len(handacc),1)),color='k',label='_nolegend_')
	plt.plot(spf*np.array(hand_inds),handacc[hand_inds],linestyle='None',marker='x',color=color_str,label='_nolegend_')
	plt.ylabel('acceleration (m/s/s)')
	plt.xlabel('time (s)')

def minDistIndex(curr_ind, handtohand):
	'''
	Purpose: 
	performs gradient descent (kind of) to find the true minimum around some guessed minimum current index of handtohand

	Inputs: 
	curr_ind - current guess at the index of a local minimum of handtohand
	handtohand - values indexed within the range of curr_ind (having a running average in the process to make this is recommended)
	
	Outputs: 
	curr_ind - updated curr_ind that is now at the nearest local minimum 
	'''
	if (curr_ind+3) >=(len(handtohand)-1) or curr_ind-3 <= 0:
		return curr_ind
	else: 
		higher = handtohand[curr_ind+3]
		lower = handtohand[curr_ind-3] 
	if higher < lower:
		multi = 1
	else: 
		multi = -1
	while higher < handtohand[curr_ind] or lower < handtohand[curr_ind]:
		#if the higher indexed version is smaller, then shift curr_ind to toward that direction until it gets larger
		if (curr_ind + multi*3 >= len(handtohand)-1) or (curr_ind + multi*3 < 0) :
			return curr_ind
		else:
			curr_ind += multi*3
			higher = handtohand[curr_ind]
	return curr_ind

def handoverSpeed(handtohand,N):
	'''
	Purpose: 
	returns handtohand velocity and acceleration each averaged with an N point convolution filter

	Inputs: 
	handtohand - ||(hand1-hand2)||_2 
	N - number of frames to do running avg over
	

	Outputs: 
	handvel = N-averaged velocity of handtohand distance measure
	handacc = N-averged acceleration of handtohand distance measure
	'''
	spf = 0.030 #seconds per frame

	handvel = handtohand[0:-1]-handtohand[1:]
	handvel = 1/spf*utils.runningAvg(handvel,N)

	handacc = handvel[0:-1]-handvel[1:]
	handacc = 1/spf*utils.runningAvg(handacc,N)

	return handvel,handacc

def getIndivJoint(user, joint_name):
	'''
	Purpose: 
	Gets the data solely associated with a particular joint names_base

	Inputs: 
	user - kinectData class object
	joint_name - string name of the joint to get data of
	

	Outputs: 
	jointxyz - m by 1 by 3 array of joint data associated with joint_name

	'''
	joint_index = user.names_base.index(joint_name)
	jointxyz = user.dataXYZ[:,joint_index,:]
	return jointxyz 

def getJointDifference(jointxyz1, jointxyz2):
	'''
	Purpose: 
	calculates the norm difference between two joints over m timesteps: ||jointxyz1_i - jointxyz2_i|| , i = 0:m-1

	Inputs: 
	jointxyz1 - m by 1 by 3 jointxyz data
	jointxyz2 - m by 1 by 3 jointxyz data, or a single 1 by 3 [x,y,z] ndarray object
	

	Outputs: 
	jointdiff - m by 1 ndarray 

	'''
	jointdiff = np.linalg.norm(jointxyz1 - jointxyz2, axis=1)
	return jointdiff

