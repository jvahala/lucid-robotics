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
	visualize.plotMovie3d(name_list,r,labels,label_labels, fig_num, title_str, note, fps)
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

	U2e 	= embed(data4,U0,feature_range=[handover_starts[1],handover_starts[2]])
	l2, c2, U2  = label(data4, feature_range=[handover_starts[2],handover_starts[3]],num_clusters=3)


	return data4,data5,handover_starts,l0,U0,l1,U1e,U1,l2,U2,U2e

if __name__ == '__main__': main()