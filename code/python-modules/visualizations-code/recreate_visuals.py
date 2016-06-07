import ../test
import ../handover_tools as ht
import ../utils
import ../assign
import matplotlib.pyplot as plt 

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

def basisPlot1d(): 
	file_name = '/Users/vahala/Desktop/Wisc/LUCID/Handover-Data/p2/p2-logs/p2-3.txt'
	data4 = test.setup('4',file_name)
	data5 = test.setup('5',file_name)
	handover_starts = ht.handoverID(data4,data5)
	l0, c0, U0  = test.label(data4, feature_range=[handover_starts[0],handover_starts[1]],num_clusters=3)
	l1, c1, U1  = test.label(data4, feature_range=[handover_starts[1],handover_starts[2]],num_clusters=3)
	l2, c2, U2  = test.label(data4, feature_range=[handover_starts[2],handover_starts[3]],num_clusters=3)
	
	u0 = U0[:,0]
	u1 = U1[:,0]
	u2 = U2[:,0]
	u0 = assign.stretchBasisCol(u0)
	u1 = assign.stretchBasisCol(u1)
	u2 = assign.stretchBasisCol(u2)

	u00 = utils.runningAvg(u0,5)
	u11 = utils.runningAvg(u1,5)
	u22 = utils.runningAvg(u2,5)

	l0 = [l+1 if l<2 else 0 for l in l0] 

	plt.figure(1)
	assign.plotClassPoints(u0,l0)
	assign.plotClassPoints(u1,l1)
	assign.plotClassPoints(u2,l2)
	
	plt.figure(2)
	assign.plotClassPoints(u00,l0)
	assign.plotClassPoints(u11,l1)
	assign.plotClassPoints(u22,l2)
	plt.title('with 5-point runningAvg')

	plt.show()

def _3dStateDists():

	file_name = '/Users/vahala/Desktop/Wisc/LUCID/Handover-Data/p2/p2-logs/p2-3.txt'
	data4 = test.setup('4',file_name)
	data5 = test.setup('5',file_name)

	handover_starts = ht.handoverID(data4,data5)
	l0, c0, U0  = test.label(data4, feature_range=[handover_starts[0],handover_starts[1]],num_clusters=3)
	l0 = [l+1 if l<2 else 0 for l in l0] 
	u0 = U0[:,0]
	
	u0 = assign.stretchBasisCol(u0)
	g0 = assign.getClassDists(u0,l0)

	g0split,counts = assign.splitStates(g0,u0,thresh_mult=1.5)
	counts0 = counts[0:3]

	#3state-dists
	plt.figure(1)
	ass.plotClassDists(g0,counts0)
	plt.yticks([])
	plt.xlabel('basis value')
	plt.ylabel('probability measure')
	plt.title('Proportional 3-state Distributions')

	#3state-dists-added
	plt.figure(2)
	ass.plotClassDists(g0split,counts)
	plt.yticks([])
	plt.xlabel('basis value')
	plt.ylabel('probability measure')
	plt.title('Proportional Added-state Distributions')

	plt.show()


def OneBasisThreeHandover(): 

	file_name = '/Users/vahala/Desktop/Wisc/LUCID/Handover-Data/p2/p2-logs/p2-3.txt'
	data4 = test.setup('4',file_name)
	data5 = test.setup('5',file_name)
	handover_starts = ht.handoverID(data4,data5)
	l0, c0, U0  = test.label(data4, feature_range=[handover_starts[0],handover_starts[1]],num_clusters=3)

	U1e = test.embed(data4,U0,feature_range=[handover_starts[1],handover_starts[2]])
	U2e = test.embed(data4,U0,feature_range=[handover_starts[2],handover_starts[3]])

	from ../kmeans import kplusplus

	l1 = kplusplus(U1e.T,k=3)
	l2 = kplusplus(U2e.T,k=3)

	l1 = utils.orderStates(l1)
	l2 = utils.orderStates(l2)
	
	u0 = U0e[:,0]
	u1 = U1e[:,0]
	u2 = U2e[:,0]
	u0 = assign.stretchBasisCol(u0)
	u1 = assign.stretchBasisCol(u1)
	u2 = assign.stretchBasisCol(u2)

	plt.figure(1)
	assign.plotClassPoints(u0,l0)
	assign.plotClassPoints(u1,l1)
	assign.plotClassPoints(u2,l2)
	plt.title('new method (basis projection)')

	plt.show()




