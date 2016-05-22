import numpy as np
import matplotlib.pyplot as plt
import utils


def handoverID(user1, user2):
	'''
	Purpose: 
	Identifies the frame numbers at which handovers occur between two human users. Handovers are defined as starting at the zero velocity point when two users hands are closest to each other. 

	Inputs: 
	user1 - kinectData class of object for user1 
	user2 - kinectData class of object for user2

	Outputs:
	start_frames - list of frame numbers associated with handover starts 

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
	N = 5
	ll = utils.runningAvg(ll,N)
	lr = utils.runningAvg(lr,N)
	rl = utils.runningAvg(rl,N)
	rr = utils.runningAvg(rr,N)

	#compute velocites of | hand1 - hand2 |, smallest velocities should be associated with extreme points 
	llvel = ll[0:-1]-ll[1:]
	lrvel = lr[0:-1]-lr[1:]
	rlvel = rl[0:-1]-rl[1:]
	rrvel = rr[0:-1]-rr[1:]
	N = 5
	llvel = utils.runningAvg(llvel,N)
	lrvel = utils.runningAvg(lrvel,N)
	rlvel = utils.runningAvg(rlvel,N)
	rrvel = utils.runningAvg(rrvel,N)

	llacc = llvel[0:-1]-llvel[1:]
	lracc = lrvel[0:-1]-lrvel[1:]
	rlacc = rlvel[0:-1]-rlvel[1:]
	rracc = rrvel[0:-1]-rrvel[1:]
	N = 5
	llacc = utils.runningAvg(llacc,N)
	lracc = utils.runningAvg(lracc,N)
	rlacc = utils.runningAvg(rlacc,N)
	rracc = utils.runningAvg(rracc,N)

	llsplit = np.logical_and(ll[0:-1] < 0.35,np.abs(llvel)<0.01)
	lrsplit = np.logical_and(lr[0:-1] < 0.35,np.abs(lrvel)<0.01)
	rlsplit = np.logical_and(rl[0:-1] < 0.35,np.abs(rlvel)<0.01)
	rrsplit = np.logical_and(rr[0:-1] < 0.35,np.abs(rrvel)<0.01)
	

	ll_inds = [ind for ind,x in enumerate(llsplit) if x==True]
	ll_inds = groupLikeIndex(ll_inds)

	lr_inds = [ind for ind,x in enumerate(lrsplit) if x==True]
	lr_inds = groupLikeIndex(lr_inds)

	rl_inds = [ind for ind,x in enumerate(rlsplit) if x==True]
	rl_inds = groupLikeIndex(rl_inds)

	rr_inds = [ind for ind,x in enumerate(rrsplit) if x==True]
	rr_inds = groupLikeIndex(rr_inds)
	
	print 'll: ', ll_inds 
	print 'lr: ', lr_inds 
	print 'rl: ', rl_inds 
	print 'rr: ', rr_inds 



	plt.subplot(3,1,1)
	plt.plot(np.arange(len(llvel)),ll[0:-1],color='r')
	plt.plot(np.arange(len(lrvel)), lr[0:-1], color='b')
	plt.plot(np.arange(len(rlvel)),rl[0:-1],color='g')
	plt.plot(np.arange(len(rrvel)), rr[0:-1], color='y')
	plt.legend(['left left','left right', 'right left', 'right right'])
	plt.ylabel('distance')
	plt.axis([180,700,0,2])
	
	plt.subplot(3,1,2)
	plt.plot(np.arange(len(llvel)),llvel,color='r')
	plt.plot(np.arange(len(lrvel)), lrvel, color='b')
	plt.plot(np.arange(len(rlvel)),rlvel,color='g')
	plt.plot(np.arange(len(rrvel)), rrvel, color='y')
	plt.plot(np.arange(len(rrvel)),np.zeros((len(rrvel),1)),color='k')
	plt.ylabel('velocity')
	plt.axis([180,700,np.amin(llvel),np.amax(llvel)])

	plt.subplot(3,1,3)
	plt.plot(np.arange(len(llacc)),llacc,color='r')
	plt.plot(np.arange(len(lracc)), lracc, color='b')
	plt.plot(np.arange(len(rlacc)),rlacc,color='g')
	plt.plot(np.arange(len(rracc)), rracc, color='y')
	plt.plot(np.arange(len(rracc)), np.zeros((len(rracc),1)),color='k')
	plt.ylabel('acceleration')
	plt.axis([180,700,np.amin(llacc),np.amax(llacc)])

	plt.show()


def groupLikeIndex(ind_list):
	temp = ind_list
	grouped = [x for ind,x in enumerate(ind_list) if (np.abs(x-temp[ind-1])>20 or ind==0)]
	return grouped




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
	jointxyz2 - m by 1 by 3 jointxyz data
	

	Outputs: 
	jointdiff - m by 1 ndarray 

	'''
	jointdiff = np.linalg.norm(jointxyz1 - jointxyz2, axis=1)
	return jointdiff


