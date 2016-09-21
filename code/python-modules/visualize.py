'''
module: visualize.py
use: contains functions associated with visualizing data, features, and joints.
'''

import numpy as np
from feature_tools import disectJoints
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def visualize(names, joints, dim,ax,color = 'k',marker='o'): 
	'''
	Purpose: 
	visualize the joint positions on a dim-dimensional platform (1-d, 2-d, or 3-d)
	NOTE: this only includes arms and mid-torso up as it stands - data set does not have the legs included

	Inputs: 
	names = base joint names returned from feature_tools.disectJoints()
	joints = n x d x k array of joints from feature_tools.disectJoints(), 
		- this function will only mess with the dim <= d columns that are input to the function
		- k > 1 if you have multiple sets of joints that you want to plot at once
	dim = dimension of space to project positions onto 

	Outputs: 
	a dim-space plot of the joint positions

	'''
	
	#set up connections array for drawing lines between joints
	#connections[i,j] = 1 means the i-th and j-th indices of possible_names are connected
	possible_names = ['ShoulderLeft', 'ShoulderRight', 'ElbowLeft', 'ElbowRight',
					'WristLeft', 'WristRight', 'HandLeft', 'HandRight', 'Head',
					'Neck', 'SpineShoulder', 'SpineMid', 'HandTipLeft', 'HandTipRight',
					'ThumbLeft', 'ThumbRight', 'HandLeftOri', 'HandRightOri'] 
	possible_names = ['ShoulderLeft', 'ShoulderRight', 'ElbowLeft', 'ElbowRight',
					'WristLeft', 'WristRight', 'HandLeft', 'HandRight', 'Head',
					'Neck', 'SpineShoulder', 'SpineMid', 'HandTipLeft', 'HandTipRight'] 

	connections = np.zeros((len(possible_names),len(possible_names)))
	adj = np.array([[0,2],[0,9],[0,10],[0,11],[1,3],[1,9],[1,10],[1,11],[2,4],[3,5],[4,6],[4,14],[5,7],
					[5,15],[6,12],[6,14],[7,13],[7,15],[8,9],[9,10],[10,11]])
	#connections[adj[:,0],adj[:,1]] = 1
	adj = np.array([[0,2],[0,9],[0,10],[0,11],[1,3],[1,9],[1,10],[1,11],[2,4],[3,5],[4,6],[5,7],[6,12],[7,13],[8,9],[9,10],[10,11]])
	#connections = utils.symmetrize(connections) #maybe include later
	joints[:,0] = -joints[:,0] #flip x axis 
	lines = np.hstack((joints[adj[:,0],0:dim],  joints[adj[:,1],0:dim]))
	
	#fig = plt.figure(fig_num)
	if dim == 2: 
		for row in lines:
			plt.plot(np.hstack((row[0],row[dim])),np.hstack((row[1],row[dim+1])),'o-'+color)
	elif dim == 3: 
		#ax = fig.add_subplot(111, projection='3d')
		for row in lines: 
			ax.plot(-1*np.hstack((row[0],row[dim])),np.hstack((row[1],row[dim+1])),np.hstack((row[2],row[dim+2])),marker=marker,color=color)
		ax.view_init(elev=-75, azim=90.1)
	#plt.show()
	#print lines.shape
	#print lines
	#print joints
	#return fig

def singlePlot3d(names_list,data_array,data_labels,label_shift, num2plot,start_plot_pos, label_labels = [''],fig_num = 1, title_str = 'Figure', note = ''): 
	fig = plt.figure(fig_num)
	ax = fig.add_subplot(111,projection='3d')
	state0,state1,state2='#7ef4cc','#0652ff','#004577'
	m0,m1,m2 = 's','o','^'
	state2 = 'k'

	for k in np.arange(num2plot):
		#print data_labels[k], k 
		if num2plot > 1: 
			names, joints = disectJoints(names_list, data_array[start_plot_pos+k,:])		
		else: 
			names, joints = disectJoints(names_list, data_array)
		if data_labels[label_shift+k] == 0: 
			color= 'r'
			color = state0
			marker = m0
		elif data_labels[label_shift+k] == 1: 
			color= 'k'
			color = state1
			marker = m1
		else: 
			color = 'y'
			color = state2
			marker = m2
		visualize(names,joints,3,ax,color,marker)
		print color
	'''
	if len(label_labels) == 2: 
		labels0 = mpatches.Patch(color='red', label=label_labels[0])
		labels1 = mpatches.Patch(color='black', label=label_labels[1])
		plt.legend(handles=[labels0,labels1],loc='lower left')
	if len(label_labels) == 3: 
		labels0 = mpatches.Patch(color=state0, label=label_labels[0])
		labels1 = mpatches.Patch(color=state1, label=label_labels[1])
		labels2 = mpatches.Patch(color=state2, label=label_labels[2])
		plt.legend(handles=[labels0,labels1,labels2],loc='lower left')
	'''
	plt.axis('tight')
	plt.plot([-100],[-100],marker=m0,color=state0,label="State 0")
	plt.plot([-100],[-100],marker=m1,color=state1,label="State 1")
	plt.plot([-100],[-100],marker=m2,color=state2,label="State 2")
	plt.legend(loc='lower-left')

	#remove tick labels and such
	ax.set_yticklabels([])
	ax.set_xticklabels([])
	ax.set_zticklabels([])
	#ax.set_xticks([])
	#ax.set_yticks([])
	#ax.set_zticks([])
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')

	plt.figtext(0.6,0.2,note,fontdict=None,fontsize=8)
	'''ax.set_xlabel()
	ax.set_ylabel()
	ax.set_zlabel()'''
	#ax.set_title(title_str, y = 0.95,fontsize = 12)
	return fig

def plotMovie3d(names_list,data_sequence,data_sequence_labels,label_labels = [''],fig_num = 1, title_str = 'Figure', note = '',fps=2):
	import os, sys

	for index,data in enumerate(data_sequence):
		names, joints = disectJoints(names_list, data)
		fig = plt.figure(fig_num)
		ax = fig.add_subplot(111,projection='3d')
		if data_sequence_labels[index] == 0: 
			color= 'r'
		elif data_sequence_labels[index] == 1: 
			color= 'k'
		elif data_sequence_labels[index] == 2: 
			color = 'y'
		elif data_sequence_labels[index] == 3: 
			color = 'm'
		elif data_sequence_labels[index] == 4: 
			color = 'b'
		else: 
			color = 'g'
		print color
		visualize(names,joints,3,ax,color)

		#pretty it up 
		if len(label_labels) == 2: 
			labels0 = mpatches.Patch(color='red', label=label_labels[0])
			labels1 = mpatches.Patch(color='black', label=label_labels[1])
			plt.legend(handles=[labels0,labels1],loc='lower left')
		elif len(label_labels) == 3: 
			labels0 = mpatches.Patch(color='red', label=label_labels[0])
			labels1 = mpatches.Patch(color='black', label=label_labels[1])
			labels2 = mpatches.Patch(color='yellow', label=label_labels[2])
			plt.legend(handles=[labels0,labels1,labels2],loc='lower left')
		elif len(label_labels) == 4: 
			labels0 = mpatches.Patch(color='red', label=label_labels[0])
			labels1 = mpatches.Patch(color='black', label=label_labels[1])
			labels2 = mpatches.Patch(color='yellow', label=label_labels[2])
			labels3 = mpatches.Patch(color='m', label=label_labels[3])
			plt.legend(handles=[labels0,labels1,labels2,labels3],loc='lower left')
		'''elif len(label_labels) == 6:
			labels0 = mpatches.Patch(color='red', label=label_labels[0])
			labels1 = mpatches.Patch(color='black', label=label_labels[1])
			labels2 = mpatches.Patch(color='yellow', label=label_labels[2])
			labels3 = mpatches.Patch(color='m', label=label_labels[3])
			labels4 = mpatches.Patch(color='b', label=label_labels[4])
			labels5 = mpatches.Patch(color='g', label=label_labels[5])
			plt.legend(handles=[labels0,labels1,labels3,labels4,labels5],loc='lower left')
		'''
	
		#remove tick labels and such
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		ax.set_zticklabels([])
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_zticks([])

		plt.figtext(0.6,0.2,note,fontdict=None,fontsize=8)
		'''ax.set_xlabel()
		ax.set_ylabel()
		ax.set_zlabel()'''
		ax.set_title(title_str, y = 0.95,fontsize = 12)
		'''
		#save and close the figure
		#frame_note = 'Frame %03d'%frame_num
		#plt.figtext(0.6,0.1,frame_note,fontdict=None,fontsize=10)'''
		fname = '_tmp%05d.png'%index
		plt.savefig(fname)
		plt.clf()
		plt.close(fig_num)

	os.system('rm movie.mp4')
	os.system('ffmpeg -framerate '+str(fps)+' -i _tmp%05d.png -s:v 500x500 -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p movie.mp4')
	os.system('rm _tmp*.png')

def plotSingleFeature(X, feature_name='Feature X', plot_range=[], color_choice = 'r', t_avg=0.030, x_unit='seconds', y_unit='meters'):
	'''
	Purpose: 
	Plot how a single feature changes with time

	Inputs: 
	X - ndarray (n examples by 1 feature)
	feature_name - name of feature to be plotted, default is 'Feature X' representing a generic feature
	plot_range - list with two elements, [start_position, stop_position], representing which of the n points in X to plot. Defaults to plot all given X. 
	color_choice - color as string for plot, default red = 'r'
	t_avg - float value representing the time rate of examples, because this module is developed for kinected data, default t_avg is 0.030 seconds representing an approximately 30fps frame rate. 
	x_unit - label for units of x axis, default is 'meters'
	y_unit - label for units of y axis, default is 'seconds'

	Outputs: 
	figure - y_axis = feature_value in y_unit units, x_axis = time in x_unit units, label = feature_name
	show figure outside of function call with plt.show() (this allows you to plot mutliple features on same plot)

	'''
	if len(plot_range) == 0: 
		x_range = np.arange(len(X))
	elif len(plot_range) == 2:
		x_range = np.arange(plot_range[0],plot_range[1])
	else: 
		return 'Error in plot_range. Supply two indices for [start_position, stop_position]'

	plt.plot(t_avg*x_range,X[x_range],marker='x',color=color_choice,label=feature_name)
	plt.xlabel('time ('+x_unit+')')
	plt.ylabel('feature value ('+y_unit+')')
	plt.legend(fontsize='x-small',numpoints=1)


