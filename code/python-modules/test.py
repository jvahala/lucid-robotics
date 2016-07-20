import kinectData
import clusters
import assign
import visualize
import utils
import numpy as np
import matplotlib.pyplot as plt
import handover_tools as ht 
import feature_tools as ft


def setup(user_id='4', file_name=''): 
	data = kinectData.kinectData(user_id) #set up kinectData object with user_id 
	if file_name == '':
		file_name = '/Users/vahala/Desktop/Wisc/LUCID/Handover-Data/p2/p2-logs/p2-3.txt'
	data.addData(file_name)
	print data.names_list, '\n shape: ', data.data_array.shape  #correct shape checked

	#uses exponential similarity array 
	data.similarity_method = 'exp'
	data.midpoint = np.array([-0.23282465, -0.23355304,  2.39738786])
	data.getFeatures()
	return data

def label(data_object, feature_range=[1000,1650], num_clusters=4, basis_dim=2, k_nn=6):

	input_data = data_object.feat_array[feature_range[0]:feature_range[1],:]

	labels, centers, U = clusters.spectralClustering(input_data, similarity_method=data_object.similarity_method, k_nn=k_nn, basis_dim=basis_dim, num_clusters=num_clusters)
	labels = utils.orderStates(labels)
	#labels = np.floor(utils.runningAvg(labels,1))
	labels = [int(x) for x in labels]
	labels = list(labels)

	#show a single frame 
	#frames_to_show = np.vstack((data.data_array[156,:],np.median(data.data_array,axis=0),data.data_array[245,:]))
	#frames_to_show = np.vstack((data.data_array[120,:],data.data_array[121,:],data.data_array[122,:],data.data_array[123,:],data.data_array[124,:],data.data_array[125,:]))
	#frames_to_show = data.data_array[100,:]
	#visualize.singlePlot3d(data.name_list,frames_to_show,[1],0,1,0,[''],1,'Skeletal Depiction','')
	#visualize.singlePlot3d(data.name_list,frames_to_show,[0,1,2],0,3,0,['Receive','Median','Place'],1,'Median Position vs. Known Receive/Place Positions','')
	#import os,sys
	#os.system('rm basic-skeleton.png')
	#plt.savefig('basic-skeleton.png')
	#visualize.singlePlot3d(data.name_list,data.data_array,labels,label_shift,num2plot,start_plot,label_labels, fig_num, title_str, note)
	#visualize.plotMovie3d(data.name_list,data.data_array[1000:1100,:],labels,label_labels, fig_num, title_str, note, fps)
	#plt.show()
	print 'labels: \n', labels
	print 'Centers: \n', centers

	print 'norm val: ', data_object.norm_value
	print 'frames: ', feature_range[0] ,'to ', feature_range[1]
	return labels, centers, U

def embed(data_object,oldU,feature_range=[100, 300],k_nn=6): 
	input_data = data_object.feat_array[feature_range[0]:feature_range[1],:]
	fullU = clusters.getLaplacianBasis(input_data,similarity_method=data_object.similarity_method,k_nn=6)
	U = utils.Subspace(oldU)
	newU = U.projectOnMe(fullU)
	return newU

def video(name_list,raw_data,labels):
	label_labels = ['Receive object', 'Mid-1', 'Mid-2', 'Place object','','']
	fig_num = 1
	title_str = 'Receiver Classification \n Six Clusters'
	note = '(no filtering/noise reduction)'
	fps = 10
	visualize.plotMovie3d(name_list,raw_data,labels,label_labels, fig_num, title_str, note, fps)
	plt.show()


def main(): 
	file_name = '/Users/vahala/Desktop/Wisc/LUCID/Handover-Data/p2/p2-logs/p2-3.txt'
	data4 = setup('4',file_name)
	data5 = setup('5',file_name)
	data4.midpoint = ft.getMidpointXYZ(data4.dataXYZ[0,:,:],data5.dataXYZ[0,:,:])
	data5.midpoint = data4.midpoint
	handover_starts = ht.handoverID(data4,data5)
	l0, c0, U0  = label(data4, feature_range=[handover_starts[0],handover_starts[1]],num_clusters=3)

	U1e 	= embed(data4,U0,feature_range=[handover_starts[1],handover_starts[2]])
	l1, c1, U1  = label(data4, feature_range=[handover_starts[1],handover_starts[2]],num_clusters=3)

	U2e 	= embed(data4,U0,feature_range=[handover_starts[2],handover_starts[3]])
	l2, c2, U2  = label(data4, feature_range=[handover_starts[2],handover_starts[3]],num_clusters=3)

	ainds = [233,257,275,293,311,315,322]
	alabs = [0,1,0,2,0,1,0]
	plot_data = data4.data_array[ainds,:]



	return data4,data5,handover_starts,l0,U0,l1,U1e,U1,l2,U2,U2e,ainds,alabs

def checkFeatures(receiver,starts):
	''' assumes receiver,giver,starts,task,curr_labels = main.begin() has been done'''
	import matplotlib.pyplot as plt 
	import assign 
	#assign.plotClassPoints(task.history[0],task.labels[0:len(task.history[0])])
	hand0 = receiver.feat_array[starts[0]:starts[1],:]
	hand1 = receiver.feat_array[starts[1]:starts[2],:]
	hand2 = receiver.feat_array[starts[2]:starts[3],:]
	hand3 = receiver.feat_array[starts[3]:starts[4],:]
	hands = [hand0, hand1, hand2, hand3]

	fig = plt.figure(num=None,figsize=(12,12))
	colors = ['k','r','b','g','y','m']
	titles = ['Feature 90:  HandRight to ThumbLeft','Feature 1:  ShoulderLeft to ElbowLeft','Feature 39:  ElbowLeft to HandTipRight','Feature 132:  HandTipLeft to Midpoint','Feature 123:  ElbowRight to Midpoint','Feature 126:  HandLeft to Midpoint']
	for i in np.arange(6):
		plt.subplot(6,1,i+1)
		for h,hand in enumerate(hands): 
			xscaled = np.arange(len(hand))*100/(1.*len(hand))
			plt.plot(xscaled,hand[:,i],color=colors[h],marker='x',linestyle='-')
		plt.title(titles[i])
		plt.xticks([])
		plt.yticks([])

	return fig 

def checkWithVels(receiver,starts,task,select=''): 
	fps = 30	#frames per second
	def iterateVelTask(curr_task_id): 
		curr_labels = []
		for item in np.arange(curr_task_id): 
			if item == 0: 
				old = getVelFeatures(item)
			else:
				addold = getVelFeatures(item)
				old = np.vstack((old,addold))
		consider = getVelFeatures(curr_task_id)

		#reshape old and consider to include traditional features as well as velocity features
		if select!='': 
			print 'traditional features added'
			old = np.hstack((old,receiver.feat_array[starts[0]:starts[curr_task_id],:]))
			consider = np.hstack((consider,receiver.feat_array[starts[curr_task_id]:starts[curr_task_id+1],:]))
		print 'old/consider,', old.shape, consider.shape
		i = 0
		for frame in consider:
			[knn_label,count_info] = utils.kNN(frame,old,task.labels, k=20) 
			curr_labels.append(knn_label)
			#print '\n\nframe number: ', i
			#print count_info
			i += 1
		return [int(x) for x in curr_labels]
	def getVelFeatures(task_id,select=''): 
		deltaT = 1/float(fps)
		relevant_feat = receiver.feat_array[starts[task_id]:starts[task_id+1]]
		#vel features will be same size as relevant features with the first element being the same as the zeroth element
		vel_feat = np.empty(relevant_feat.shape)
		vel_feat[1:,:] = (relevant_feat[0:-1,:] - relevant_feat[1:,:])/deltaT
		vel_feat[0,:] = vel_feat[1,:]
		return vel_feat
	print '\nVelocity Features: \n--------------------\n'
	curr_labels = iterateVelTask(1)
	print curr_labels



		







if __name__ == '__main__': main()