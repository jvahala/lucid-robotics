import process
import assign
import utils
import kinectData
import clusters
import numpy as np

def setupComparison(data1,data2):
	inds = data1.feature_inds 
	print data2.feat_array[0,:]
	data2.feat_array = data2.all_features[:,inds]
	print data2.feat_array[0,:]
	data2.feature_inds = inds

def getSeveralBases(data,curr_extrema,several=11):
	def defineInitBasisLabels(data,curr_extrema,basis_dim, k): 
		curr_input = data.feat_array[curr_extrema[0]:curr_extrema[1],:]
		labels, centers, U = clusters.spectralClustering(curr_input, similarity_method=data.similarity_method, k_nn=6, basis_dim=basis_dim, num_clusters=k)
		labels = utils.orderStates(labels)
		labels = [int(x) for x in labels]
		labels = list(labels)
		for ind,col in enumerate(U.T): 
			U[:,ind] = assign.stretchBasisCol(col)
		return labels, U
	labels,U = defineInitBasisLabels(data,curr_extrema,basis_dim=several,k=3)
	for i in range(several):
		assign.plotClassPoints(U[:,i],labels)


def combineTasks(data1,data2,starts,howmany):
	task1 = process.Task(data_object=data1,curr_extrema=[starts[0],starts[1]],k=3)
	task2 = process.Task(data_object=data2,curr_extrema=[starts[0],starts[1]],k=3)
	task1init = [task1.path, task1.times]
	task2init = [task2.path, task2.times]

	assign.plotClassPoints(task1.history[0],task1.labels)

	print 'task1:', task1init
	print 'task2:', task2init
	
	for i in range(howmany):
		#task1.update(data1,[starts[i+1],starts[i+2]])
		task1.update(data2,[starts[i+1],starts[i+2]])
		task2.update(data1,[starts[i+1],starts[i+2]])
		#task2.update(data2,[starts[i+1],starts[i+2]])

	return task1,task2

def plotTaskBases(task):
	start = 0
	for h in task.history: 
		assign.plotClassPoints(h,task.labels[start:start+len(h)])
		start += len(h)

def setup(task_id='p2-3'): 
	from main import begin 
	receiver,giver,starts,rtask,gtask = begin(task_id)
	print 'here'
	print giver.feat_array[0,:]
	setupComparison(receiver,giver)
	print giver.feat_array[0,:]
	print 'there'
	return receiver,giver,starts 

def getSubspaces(receiver,giver,starts,dim=4):
	urec = {}
	ugiv = {}
	for i in range(11): 
		task = process.Task(receiver,[starts[i],starts[i+1]],k=3,basis_dim=dim)	#gets 5 different basis vectors
		urec[i] = task.subspace
		task = process.Task(giver,[starts[i],starts[i+1]],k=3,basis_dim=dim)
		ugiv[i] = task.subspace
	return urec,ugiv 

def compareBases(subspace,Unew): 
	z = subspace.projectOnMe(Unew,onlyshape=True)		#get corrected shape of new task 
	error = []
	for i,col in enumerate(z.T): 
		error.append(np.sum(np.abs(subspace.U[:,-2]-col)))
	return z,error 

def subspaceTimeWarp(subspace1,subspace2,col1,col2,constraint=0.05 ,window=10): 
	# reshape second subspace basis to fit length of first subspace
	C = subspace1.projectOnMe(subspace2.U,onlyshape=True)

	#define time series 
	q = subspace1.U[:,col1]
	#print 'info: ', subspace2.U[0,-1], subspace1.U[0,-1], int(subspace2.U[0,-1]) == int(subspace1.U[0,-1])
	if int(subspace2.U[0,-1]) != int(subspace1.U[0,-1]): 
		print 'made it'
		C = -1*C
	c = C[:,col2]
	#print 'lenghts: ', len(q),len(c)

	c = utils.runningAvg(c,window)
	q = utils.runningAvg(q,window)

	path,cost = basisTimeWarp(q,c,constraint)

	return q,c,path,cost
	
def basisTimeWarp(q,c,constraint=0.05 ,dist='squared'): 

	qlen,clen = len(q),len(c)
	#print 'qlen,clen', qlen,clen

	#define distance matrix D 
	D = np.zeros((qlen,clen))
	for i in np.arange(qlen):
		for j in np.arange(clen):
			if dist=='squared': 
				D[i,j] = (q[i]-c[j])**2
			else: 
				D[i,j] = np.abs(q[i]-c[j])

	#dynamic programming
	i = qlen-1
	j = clen-1

	path = [[i,j]]		#final point will always happen
	cost = 0

	#define cost map
	G = np.zeros((qlen,clen))		
	G[i,j] = D[i,j]
	for k in np.arange(1,i+1):		#can only go one direction if at end point for i or j
		G[i-k,j] = np.sum(G[(i-k+1):,j]) + D[i-k,j]
	for n in np.arange(1,j+1):
		G[i,j-n] = np.sum(G[i,(j-n+1):]) + D[i,j-n]
	#fill in the rest via G(i,j) = D(i,j) + min{G(i-1,j-1),G(i-1,j),G(i,j-1)}
	for k in np.arange(1,i+1): 
		for n in np.arange(1,j+1): 
			G[i-k,j-n] = min(G[i-k+1,j-n],G[i-k+1,j-n+1],G[i-k,j-n+1]) + D[i-k,j-n]

	#determine path and accumated cost 
	while i>0 and j>0:
		if i==0: 
			j -= 1
		elif j==0: 
			i -= 1
		elif np.abs(i-j) > constraint*qlen: 		#i and j are too different
			if i > j: 
				minG = min(G[i-1,j],G[i-1,j-1])
				if G[i-1,j] == minG: 
					i -= 1
				elif G[i-1,j-1] == minG: 
					i -= 1
					j -= 1
			elif i < j: 
				minG = min(G[i,j-1],G[i-1,j-1])
				if G[i,j-1] == minG: 
					j -= 1
				elif G[i-1,j-1] == minG: 
					i -= 1
					j -= 1
		else:
			minG = min(G[i-1,j],G[i-1,j-1],G[i,j-1])
			if G[i-1,j] == minG: 
				i -= 1
			elif G[i,j-1] == minG: 
				j -= 1
			else: 
				i = i-1
				j = j-1
		path.append([i,j])
	path.append([0,0])
	for [x,y] in path: 
		cost = cost + D[x, y]
	return path,cost

def plotit(q,c,path): 
	import matplotlib.pyplot as plt
	plt.plot(q, 'bo-' ,label='q')
	plt.plot(c, 'g^-', label = 'c')
	plt.legend();
	for [map_q, map_c] in path:
		print map_q, q[map_q], ":", map_c, c[map_c]
		plt.plot([map_q, map_c], [q[map_q], c[map_c]], 'r')

def getCosts(subspace1,subspace2,col1,constraint=0.75,window=3):
	costs = [] 
	for col2 in range(subspace2.U.shape[1]): 
		q,c,path,cost = subspaceTimeWarp(subspace1,subspace2,col1,col2,constraint,window)
		print 'column: ', col2, 'cost: ', cost
		costs.append(cost)
	return costs
	
def printMinCosts(subspace1,subspace2_group,col1=2,constraint=0.75,window=3):
	costs_list = []
	for i in range(11): 
		costs = getCosts(subspace1,subspace2_group[i],col1,constraint,window)
		costs_list.append(np.mean(costs[:-1]))
	print '\n\nAvg Costs\n'
	for item in costs_list: 
		print item 
	return costs_list

def compareFeatureCosts(data_obj1,data_obj2,starts1,starts2,feature,numtasks=10,constraint=0.05):
	'''
	Purpose: 
	creates a grid of task to task comparisons colored based on the strength of the costs from DTW. One half of the diagonal represents the inter-task-type comparisons (ie receiver task 0 vs receiver task 1), and the alternate half of the diagonal reprsents the contra-task-type comparisons (ie receiver task 0 vs giver task 1)

	'''
	#define all the task bases 
	feature_inds = data_obj1.feature_inds
	data1_features = data_obj1.all_features[:,feature_inds]
	data2_features = data_obj2.all_features[:,feature_inds]	#its a dumb name, semi-ignore the name
	relevant_feature = data1_features[:,feature]
	irrelevant_feature = data2_features[:,feature]
	R = {}
	G = {}
	for i in np.arange(numtasks):
		ri = relevant_feature[starts1[i]:starts1[i+1]]
		gi = irrelevant_feature[starts2[i]:starts2[i+1]]
		R[i] = ri.reshape(len(ri),1)
		G[i] = gi.reshape(len(gi),1)

	costmap = np.zeros((numtasks,numtasks))
	for i,r in R.iteritems(): 
		for j,rc in R.iteritems(): 
			if j>i: 
				path,cost = basisTimeWarp(r,rc,constraint=constraint)
				costmap[i,j] = cost 
		for k,g in G.iteritems(): 
			if k>=i: 
				path,cost = basisTimeWarp(r,g,constraint=constraint)
				costmap[k,i] = cost

	return costmap 

def plotCostMap(data_obj1,data_obj2,starts1,starts2,endtype='median',numtasks=10,constraint=0.05,colormap='cool',threshold=True,plot=True):
	import matplotlib.cm as cm 
	import matplotlib.pyplot as plt 
	my_cmap = cm.get_cmap(colormap)

	num_features = 6
	costmaps = ['']*num_features 
	for i in range(num_features): 
		costmaps[i] = compareFeatureCosts(data_obj1,data_obj2,starts1,starts2,i,numtasks,constraint)
		print 'feature', i, 'complete'
	if endtype == 'avg':
		costmap = np.mean(costmaps,axis=0)
	elif endtype == 'median': 
		costmap = np.median(costmaps,axis=0)
	if threshold: 
		for i,c in enumerate(costmap): 
			for j,k in enumerate(c): 
				if k > 3.353: 
					costmap[i,j] = 12
				elif k<=3.353: 
					costmap[i,j] = 4

	x = np.arange(numtasks+1)
	X,Y = np.meshgrid(x,x)

	if plot: 
		plt.pcolor(X,Y,costmap,cmap=my_cmap)
		cbar = plt.colorbar(ticks=[])
		#plt.gca().invert_yaxis()
		plt.title('Task Comparison Matrix - receiver vs. giver tasks')
		plt.yticks(np.arange(numtasks)+0.5,range(numtasks))
		plt.xticks(np.arange(numtasks)+0.5,range(numtasks))
		plt.ylabel('Comparison Task Number')
		plt.xlabel('Base Task Number')

	print costmap 
	return costmap 
				


def getTaskMetric(path,times,current_labels,position,frames_since_state_change,constraint=0.05):
	estimated_path = createPathObject(path,times,position,frames_since_state_change)
	estimated_path = np.array(estimated_path).reshape(len(estimated_path),1)
	current_labels = np.array(current_labels).reshape(len(current_labels),1)
	#print 'est shape: ', estimated_path.shape
	#print 'curr shape:', current_labels.shape
	path,cost= basisTimeWarp(estimated_path,current_labels,constraint=constraint)
	return cost

def createPathObject(path,times,position,frames_since_state_change): 
	full = []
	for i,p in enumerate(path): 
		if i < position: 
			full += [p]*times[i]
		else: 
			full += [p]*min(times[i],frames_since_state_change)
			break
	return full

def getAllCostmaps(txtnames,kinectDict,startsDict,numtasks=8):
	all_costmaps = {}
	for i in np.arange(len(txtnames)):
		for j in np.arange(len(txtnames)):
			if i == j: 
				continue
			type1,type2 = txtnames[i],txtnames[j]
			ij = str(i)+str(j)
			all_costmaps[ij] = plotCostMap(kinectDict[type1],kinectDict[type2],startsDict[type1],startsDict[type2],numtasks=numtasks,threshold=False,plot=False)
	return all_costmaps

def getMedianComparisons(costmap):
	same_vals,diff_vals = [],[]
	for i,row in enumerate(costmap):
		for j,item in enumerate(row):
			if j > i:
				same_vals.append(item)
			else: 
				diff_vals.append(item)
	median_same = np.median(same_vals)
	median_diff = np.median(diff_vals)
	return median_same,median_diff

def getMedianComparisonsDict(all_costmaps):
	comparison_dict = {}
	for k,v in all_costmaps.iteritems():
		same,diff = getMedianComparisons(v)
		comparison_dict[k] = [same, diff]
	return comparison_dict

def getGammaFitRV(data):	# deprecated due to seaborn being awesome
	import scipy.stats as stats 
	fit_alpha,fit_loc, fit_beta = stats.gamma.fit(data)
	rv = stats.gamma(fit_alpha,fit_loc,fit_beta)
	return rv 

def plotGammaRVs(datas,labels): 
	import matplotlib.pyplot as plt
	for i,data in enumerate(datas): 
		max_val = np.amax(data)
		x = np.linspace(0,30,100)
		rv = getGammaFitRV(data)
		y = rv.pdf(x)*len(data)
		plt.plot(x,y,'-',label=labels[i])
	plt.legend()
	plt.grid(True)

def histAndGammaPlot(datas,labels):
	import seaborn as sns 
	import scipy.stats as stats
	import matplotlib.pyplot as plt
	fig = plt.figure(1)
	ax = plt.subplot(111)
	fit_intersect = [1.79,0.062]
	sns.set_style(style='whitegrid',rc={"lines.linewidth": 0.5})
	sns.set_context(context="paper")
	sns.color_palette(palette="Paired")
	sns.distplot(datas[0],kde=False,fit=stats.gamma,label='Same')
	sns.distplot(datas[1],kde=False,fit=stats.gamma,label='Different')
	sns.despine(trim=True)
	plt.legend()
	ax.annotate('1.79', xy=(fit_intersect[0], fit_intersect[1]), color='gray',xytext=(7, 0.65),arrowprops={'arrowstyle': '-|>','color':'gray'})
	#ax.text(fit_intersect[0],fit_intersect[1]+0.03,fit_intersect[0],color='gray')
	plt.title('Intertask Comparison by Median DTW Costs')
	plt.xlabel('Dynamic Time Warping Cost')

	plt.show()



def main(): 
	pass

















if __name__== '__main__': main()