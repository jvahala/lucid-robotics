import test
import handover_tools as ht
import utils

def handToHandStats():
	user1 = '4'
	user2 = '5'
	filename = '/Users/vahala/Desktop/Wisc/LUCID/Handover-Data/p2/p2-logs/p2-3.txt'
	data1 = test.setup(user1,filename)
	data2 = test.setup(user2,filename)

	joint_left = 'HandLeft'
	left1 = ht.getIndivJoint(user1,joint_left)
	left2 = ht.getIndivJoint(user2,joint_left)

	joint_right = 'HandRight'
	right1 = ht.getIndivJoint(user1,joint_right)
	right2 = ht.getIndivJoint(user2,joint_right)

	#create new variables associated with all difference potential handovers (user 1 left hand to user 2 right hand, etc)
	ll = ht.getJointDifference(left1,left2)
	lr = ht.getJointDifference(left1,right2)
	rl = ht.getJointDifference(right1,left2)
	rr = ht.getJointDifference(right1,right2)
	N = 10
	ll = utils.runningAvg(ll,N)
	lr = utils.runningAvg(lr,N)
	rl = utils.runningAvg(rl,N)
	rr = utils.runningAvg(rr,N)

	#compute velocites of | hand1 - hand2 |, smallest velocities should be associated with extreme points 

	llvel, llacc = ht.handoverSpeed(ll,N)
	lrvel, lracc = ht.handoverSpeed(lr,N)
	rlvel, rlacc = ht.handoverSpeed(rl,N)
	rrvel, rracc = ht.handoverSpeed(rr,N)

	min_dist = 0.4	#meters
	min_vel = 0.5	#meters/sec
	llsplit = np.logical_and(ll[0:-1] < min_dist, np.abs(llvel)<min_vel)
	lrsplit = np.logical_and(lr[0:-1] < min_dist, np.abs(lrvel)<min_vel)
	rlsplit = np.logical_and(rl[0:-1] < min_dist, np.abs(rlvel)<min_vel)
	rrsplit = np.logical_and(rr[0:-1] < min_dist, np.abs(rrvel)<min_vel)
	

	ll_inds = [ind for ind,x in enumerate(llsplit) if x==True]
	ll_inds = ht.groupLikeIndex(ll_inds,len(llacc))
	ll_inds = [ht.minDistIndex(x,ll) for x in ll_inds]

	lr_inds = [ind for ind,x in enumerate(lrsplit) if x==True]
	lr_inds = ht.groupLikeIndex(lr_inds,len(lracc))
	lr_inds = [ht.minDistIndex(x,lr) for x in lr_inds]

	rl_inds = [ind for ind,x in enumerate(rlsplit) if x==True]
	rl_inds = ht.groupLikeIndex(rl_inds,len(rlacc))
	rl_inds = [ht.minDistIndex(x,rl) for x in rl_inds]

	rr_inds = [ind for ind,x in enumerate(rrsplit) if x==True]
	rr_inds = ht.groupLikeIndex(rr_inds,len(rracc))
	rr_inds = [ht.minDistIndex(x,rr) for x in rr_inds]
	
	handover_starts = ll_inds+lr_inds+rl_inds+rr_inds
	print 'starts', handover_starts
	plt.figure(1)
	ht.plotHandoverInfo(ll,llvel,llacc,ll_inds,color_str='r')
	ht.plotHandoverInfo(lr,lrvel,lracc,lr_inds,color_str='b')
	ht.plotHandoverInfo(rl,rlvel,rlacc,rl_inds,color_str='g')
	ht.plotHandoverInfo(rr,rrvel,rracc,rr_inds,color_str='y')
	plt.legend(['left left','left right', 'right left', 'right right'])


	plt.show()