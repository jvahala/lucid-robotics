'''
module: feature_tools.py 
use: contains functions associated with messing with generating features from joint data 
'''

import numpy as np
from utils import getDifferenceArray
from utils import getSimilarityArray
from utils import normalize

np.set_printoptions(threshold=np.nan,precision=3,suppress=True)

def disectJoints(names, joints): #takes 1xn array vector of joint positions named (joint.x, joint.y, joint.z, etc) and turns them into individual joint position vectors (currently ignores HandOriLeft and HandOriRight)
	'''
	Purpose:
	Takes the names and joint values from kinect data, and turns them into an N by 3 (x,y,z) array for easier use in other functions
	Does not use Left/RightHandOri.W  

	Inputs: 
	names - name list of the values in the joints array (hand.X, shoulder.Y, etc)
	joints - kinect joint data associated with each element in names

	Outputs: 
	names3d - the N joint names (less the .X, .Y, .Z) as a list, one per base joint type (ie LeftShoulder)
	joints3d - N by 3 (x,y,z) ndarray representing the components of each base joint type described in names3d

	'''
	num_joints = len(joints)
	name_base = []
	for name in names: 
		name_base.append(name[:-2])
	base_names,base_ind,base_inv,base_cnt = np.unique(name_base,return_index=True,return_inverse=True,return_counts=True) #remove the .X, .Y, etc to get base names and the number of each base name
	joints3d = np.zeros((1,3))
	names3d = []
	ind = 0
	for i in range(0,len(base_names)):
		base = base_ind[base_inv[ind]]
		next_ind = ind+base_cnt[base_inv[ind]]
		if base_names[base_inv[ind]] not in ['HandLeftOri','HandRightOri']: #ignore these features
			joints3d = np.vstack((joints3d,np.array([joints[ind],
				 								joints[ind+1],
				 								joints[ind+2]])))
			names3d.append(base_names[base_inv[ind]])
		ind = next_ind
	return names3d, joints3d[1:,:]

def getMidpointXYZ(user1_joints,user2_joints): #returns the midpoint of two joints3d arrays representing player1 and player2's [x,y,z] corrdinates for the various joints
	'''
	Purpose: 
	Returns the midpoint (x,y,z) position between two averaged user joint arrays

	Inputs: 
	user1_joints - N joints by 3 dim (x,y,z) ndarray for user1 
	user2_joints - N joints by 3 dim (x,y,z) ndarray for user2

	Outputs: 
	(avg_user1+avg_user2) / 2.0 

	'''
	avg_user1 = np.array([np.mean(user1_joints[:,0]),np.mean(user1_joints[:,1]),np.mean(user1_joints[:,2])])
	avg_user2 = np.array([np.mean(user2_joints[:,0]),np.mean(user2_joints[:,1]),np.mean(user2_joints[:,2])])
	return (avg_user1+avg_user2)/2.0

def describeFeatures(feature_index,num_joints,names_list=[]):
	#input a list of the feature indices, the number of joints you are considering as data, and optionally the names_list of the joints

	#get interjoint length from number of joints 
	interjoint_length = (num_joints**2 - num_joints)/2
	names = names_list
	if num_joints == len(names_list) and names_list[-1] == 'Midpoint': 
		names = names_list[:-1]
		num_joints = num_joints -1
		interjoint_length = (num_joints**2 - num_joints)/2
	#print interjoint_length

	#create dummy array for interjoint_features, the index equal to the feature_index will be the two joints involved
	f = np.arange(interjoint_length)
	dummy = np.ones((num_joints,num_joints))*-1
	for i in np.arange(0,num_joints-1):
		dummy[i,i+1:] = f[0:num_joints-1-i]
		f = f[num_joints-i-1:]
	indices = []
	feat_ind = 0
	for feature in feature_index:
		if feature < interjoint_length: 
			ind = (np.where(dummy==feature))
			#print ind
			indices.append(ind)
		else: 
			ind = (np.array([np.int(feature-interjoint_length)]),np.array([len(names)]))
			indices.append(ind)
	#print out in text the associated joints
	if len(names)>0: 
		if names[-1] != 'Midpoint':
			names.append('Midpoint')
		for x in indices: 
			print 'Feature %d:  %s to %s'%(feature_index[feat_ind],names[x[0]],names[x[1]])
			feat_ind += 1
	#print indices
	#print dummy
	return indices

def getInterjointFeatures(joints): #input the difference array, return the corresponding vector of features
	'''
	Purpose: 
	Returns the features associated with the interjoint distances of, presumably, a single person

	Inputs: 
	joints - N joints by 3 dim (X,Y,Z) array retrieved through disectJoints()

	Outputs: 
	feature_vector - 1 by L array described by the difference array which caluclates the norm between each joint with all other joints (self excluded)

	'''
	difference_array = getDifferenceArray(joints)
	k = 1 
	j = 0
	feature_vector = np.zeros(1)
	elmnt = []
	for i in range(0,len(difference_array)): 
		feature_vector = np.hstack((feature_vector,difference_array[i,k:len(difference_array)]))
		k += 1
	#print 'num interjoint: ', feature_vector.shape
	return feature_vector[1:len(feature_vector)] #return feature vector without the leading 0 from initialization

def getJointToMidpointFeatures(joints,midpoint): 
	'''
	Purpose: 
	Returns the features between joints and midpoints: feature_i = ||joint_i - midpoint|| 

	Inputs: 
	joints - N joints by 3 dim (X,Y,Z) array retrieved through disectJoints()
	midpoint - midpoint between two sets of joints (presumably different people's joints at the same time) retrieved through getMidpointXYZ

	Outputs: 
	features - 1 by L array described by feature_i = ||joint_i - midpoint||

	'''
	#input joints[nx3] (from function disectJoints()) and midpoint[1x3] between two users (from function getMidpointXYZ())
	#return feature vector features with features_i = ||joint_i - midpoint||^2
	features = np.linalg.norm(joints-midpoint,axis=1)
	#print 'midpoint features: ', features.shape
	return features

def thresholdFeatures(features,norm_value): 
	'''
	Purpose: 
	Defined a new set of features that are time_varying soas to lower the complexity
	general idea: REWRITE WHAT IS GOING ON HERE

	Inputs: 
	features - set of n by k features as an ndarray
	norm_value - normalizing value for the user (from kinectData.norm_value)

	Outputs: 
	new_features - n by p<k set of the most important p features in the original feature array
	feature_inds - indices of the features that were chosen 
	'''
	features0_1 = features[:,:]/(1.*np.amax(features,axis=0)) #normalize given feature set so thresholds in this function make sense

	#determine feature groups by comparing how features change together
	#names, names_len needed
	names = ['ShoulderLeft', 'ShoulderRight', 'ElbowLeft', 'ElbowRight', 'WristLeft', 'WristRight', 'HandLeft', 'HandRight', 'Head', 'Neck', 'SpineShoulder', 'SpineMid', 'HandTipLeft', 'HandTipRight', 'ThumbLeft', 'ThumbRight']
	#print names, len(names), features.T.shape[0]
	all_feat = np.arange(features.T.shape[0])
	names_len = sum(1 for i in names if i!='Midpoint')
	#describeFeatures(list(all_feat),len(names),names)

	'''no grouping but with threshold
	#get midpoint of dataset
	#compare each point to midpoint and sum up differences, get 15 best in order
	median_nogroup = np.median(features0_1,axis=0)
	power_nogroup = np.sum(np.abs(features0_1 - median_nogroup),axis=0)
	power_nogroup = normalize(power_nogroup,np.amax(power_nogroup))
	power_inds_nogroup = [x for x in np.argsort(power_nogroup)[-15:][::-1]]
	print 'Power: \n', power_inds_nogroup
	describeFeatures(power_inds_nogroup,len(names),names)'''



	groups = -1*np.ones((1,len(names)))
	i = 0

	#for each feature, compare with the 16 other features that share the same name
	#then, jump to next key feature joint and repeat
	skip_to_next = False
	groups_row = 0
	amount = len(names)-3
	start = amount
	norm_diff = -1*np.ones(len(names))
	for ind1, feat in enumerate(features0_1[:,0:-names_len+1].T):
		groups_col = 0
		if ind1 > start: 
			if ind1 > len(names):
				amount = amount - 1
				if amount == 0: 
					amount = len(names)-1
				start += amount
			else: 
				start += amount
			skip_to_next = False
		curr_norm = -1*np.ones(len(names))
		curr_group = -1*np.ones(len(names))
		for ind2, feat2 in enumerate(features0_1.T):
			if ind2>start: 
				skip_to_next = True
				break
			if ind2 < ind1 or ind2 < start-amount:
				continue
			#print '\tind2: ',ind2
			diff = np.linalg.norm(feat-feat2)
			position = (ind2-ind1)%(amount+1)
			curr_norm[position] = np.exp(-1.*diff)
			curr_group[position] = ind2
		norm_diff = np.vstack((norm_diff,curr_norm))
		groups = np.vstack((groups,curr_group))

	norm_diff = norm_diff[1:,:] #remove initalization row, this is the norm-ed differences between features
	groups = groups[1:,:]	#remove intitialization row, row i = the base feature, items in groups = the features within that comparison group


	#Define a threshold strength for the norm_diff values that mean two features should be consolidated
	thresh_strength = 0.2 #larger values (up to 1) mean the features must be more correlated, 0.2 is fine as uncorrelated features quickly get small in this strength measure
	
	#Build initial group of feature sets that exceed the thresh_strength
	strong_groups = []
	for ind, group in enumerate(groups): 
		if group.all() == groups[-1].all():
			thresh_strength = 0
		listing = group[0:np.argmax(group == -1)]
		listing = listing[norm_diff[ind,0:np.argmax(norm_diff[ind,:] == -1)]>=thresh_strength]
		feature_inds = list(listing)
		strong_groups.append(feature_inds)
	
	#Consoldiate initial groupings  
	#strong_groups is a num_features length list of lists representing each feature's grouping
	new_groups = []
	#define end points for each gruop
	stop_points = [14]
	for i in np.arange(14): 
		stop_points.append(stop_points[i]+14-i)
	#check amoung each each primary joint group for closed connection maps
	#ie if groups 1:6 represent Feature1 as the primary and they look like [[1,2],[2],[3,1],[4,5],[5],[6]]
	#they will consolidate to [[1,2,3],[4,5],[6]] meaning only three features maximum will need to be used for that set instead of six
	for stop_ind, stop in enumerate(stop_points[0:]): 
		if stop_ind == 0: 
			start = 0
		else: 
			start = stop_points[stop_ind-1]+1
		moved = []
		for indgi, gi in enumerate(strong_groups[start:stop]): 
			for ind,item in enumerate(gi): 
				remove_these = []
				if item in moved: 
					for indgi2, gi2 in enumerate(strong_groups[start:stop]): 
						if item in gi2: 
							move_to = indgi2 + start
							break
					for item2 in gi: 
						if item2 not in strong_groups[move_to] and item2<item and gi2 != gi:
							#print 'hitagain'
							strong_groups[move_to].append(item2)
							remove_these.append(item2)
						elif item2 in strong_groups[move_to] and gi2!=gi: 
							remove_these.append(item2)
					for remove_it in remove_these: 
						strong_groups[np.int(remove_these[0])].remove(remove_it)
					remove_these = []
					#break
				for subitem in strong_groups[np.int(item)]:
					if subitem in gi: 
						if item != indgi and item > indgi+start: 
							remove_these.append(subitem)
						continue
					gi.append(subitem)
					moved.append(subitem)
					remove_these.append(subitem)
				for remove_it in remove_these: 
					strong_groups[np.int(item)].remove(remove_it)
				remove_these = []
	strong_groups = [x for x in strong_groups if x] #remove empty lists from strong groups list


	#print the new strong groups for clarity sake - not critical to function 
	'''for ind, group in enumerate(strong_groups): 
		print '\nFinal GROUP %d'%ind
		feature_inds = group
		print feature_inds
		describeFeatures(feature_inds, len(names), names)'''

	#select first element of each group (will be consistent in future calls to this function?) as representative group (doesn't matter)
	inter_test_set = [group[0] for group in strong_groups[0:-1]] #interjoint test set
	midpt_test_set = strong_groups[-1] #select whole of the joint to midpoint features as candidates
	#print midpt_test_set
	
	#see how these features change with the data_set sent in
	inter_test_features = features0_1[:,inter_test_set]
	midpt_test_features = features0_1[:,midpt_test_set]

	#test_features_mean = np.mean(test_features,axis=0) #get mean for each column
	inter_test_med = np.median(inter_test_features,axis=0) #get median for each column 
	midpt_test_med = np.median(midpt_test_features,axis=0)

	#aggreate the difference of each feature from its median
	inter_diff_med = np.sum(np.abs(inter_test_features-inter_test_med),axis=0)
	inter_diff_med = normalize(inter_diff_med,np.amax(inter_diff_med))

	midpt_diff_med = np.sum(np.abs(midpt_test_features-midpt_test_med),axis=0)
	midpt_diff_med = normalize(midpt_diff_med,np.amax(midpt_diff_med))



	#need to separate here into groups of feature types - interjoint vs joint-to-midpoint, 
	#then select some number above a threshold from each

	#interjoint features
	#inter_top_med gives the most changing features when compared to their median values in order of the most changing feature group to the least changing. The following code then adds features to the list of relevant features if they share at least some dissimilarity to the previously added features
	num_select = 3*names_len #choose a number related to how many joints you have
	inter_top_med = [x for x in np.argsort(inter_diff_med)[-num_select:][::-1]] #give largest index first

	#compare the features to eachother again and take only those that are dissimilar
	sim_array_temp = getSimilarityArray(features0_1[:,inter_top_med],'exp',-1)
	#print sim_array_temp
	'''want to change this sim_threshold to allow at least some number of features, maybe can do this by simply adding the feature most dissimilar from the group until k items are reached.'''
	sim_threshold = 0.8 #threshold between 0 and 1. Larger thresholds mean more similarity between features is allowed
	#if next one is greater than some similarity threshold, of the previous ones, do not put it in
	'''get min(similary[feature in index_set vs. all inter_features]), then add to group, then for each of the features in index set, sum up all the distances and get the min similarity again, will want to do this until the sum of the similarities is very high or something ... implement this for the midpoint features as well...maybe its better to look for the most medium similarity feature instead of the most dissimilar/ '''

	#set up first inter_feature, index_set feature as the  most changing feature group
	inter_features = [strong_groups[inter_top_med[0]][0]]
	index_set = [0] 

	#while some criteria isnt met, add features until it is 
	while len(index_set)<3: 
		keys = [int(x) for x in np.arange(len(inter_top_med)) if x not in index_set]
		values = np.zeros(len(keys))
		choices = dict(zip(keys,values))
		for i in np.arange(len(inter_features)): 
			for ind,f in enumerate(inter_top_med): 
				if ind in index_set: 
					continue
				key = ind		#because ind will start at 0
				choices[key] += sim_array_temp[index_set[i],key]		#for each of the inter top med indices, add the value of the similiarity array at the location
		med_choice_value = np.median(choices.values())
		med_choice_value_diffs = [v-med_choice_value for v in choices.values()]
		choice_key = choices.keys()[np.argmin(med_choice_value_diffs)]
		inter_features.append(strong_groups[inter_top_med[choice_key]][0])
		index_set.append(choice_key)	#adds the most different feature 


	'''for ind, f in enumerate(inter_top_med): 
		if ind == 0: 
			inter_features = [strong_groups[f][0]]
			index_set = [ind]
		else: 
			for i in np.arange(len(inter_features)):
				if sim_array_temp[index_set[i],ind]<sim_threshold:
					if i == len(inter_features)-1:
						inter_features.append(strong_groups[f][0])
						index_set.append(ind)
					else: 
						continue
				else:
					break
		if len(inter_features)>names_len: #don't allow more features than the number of joints
			break '''

	#do same kind of thing with midpoint features
	midpt_top_med = [x for x in np.argsort(midpt_diff_med)[::-1]] #give largest index first
	print 'midpoint top med init; ', midpt_top_med

	#compare the features to eachother again and take only those that are dissimilar
	sim_inds = [midpt_test_set[i] for i in midpt_top_med]
	print 'sim_inds: ',  sim_inds
	#print sim_inds
	sim_array_temp = getSimilarityArray(features0_1[:,sim_inds],'exp',-1)
	#print sim_array_temp
	sim_threshold = 0.99 #threshold between 0 and 1. Larger thresholds mean more similarity between features is allowed
	''' this section here is the new stuff... it current gives some not so good feautures'''
		#set up first inter_feature, index_set feature as the  most changing feature group
	midpt_features = [sim_inds[0]]
	index_set = [0] 

	#while some criteria isnt met, add features until it is 
	while len(index_set)<3: 
		keys = [int(x) for x in np.arange(len(midpt_top_med)) if x not in index_set]
		values = np.zeros(len(keys))
		choices = dict(zip(keys,values))
		for i in np.arange(len(midpt_features)): 
			for ind,f in enumerate(midpt_top_med): 
				if ind in index_set: 
					continue
				key = ind		#because ind will start at 0
				choices[key] += sim_array_temp[index_set[i],key]		#for each of the midpt top med indices, add the value of the similiarity array at the location
		med_choice_value = np.median(choices.values())
		med_choice_value_diffs = [v-med_choice_value for v in choices.values()]
		choice_key = choices.keys()[np.argmin(med_choice_value_diffs)]
		#choice_key = next((k for k,v in choices.iteritems() if v == int(np.median(choices.values()))),None)
		#choice_key = min(choices,key=choices.get)
		midpt_features.append(sim_inds[choice_key])
		index_set.append(choice_key)	#adds the most different feature 
		print 'STEP: ', len(index_set)-1, index_set, midpt_features

	'''this commented thing is the old stuff'''
	#if next one is greater than some similarity threshold, of the previous ones, do not put it in
	'''for ind, f in enumerate(midpt_top_med): 
		if ind == 0: 
			midpt_features = [midpt_test_set[f]]
			index_set = [ind]
		else: 
			for i in np.arange(len(midpt_features)):
				if sim_array_temp[index_set[i],ind]<sim_threshold:
					if i == len(midpt_features)-1:
						midpt_features.append(midpt_test_set[f])
						index_set.append(ind)
					else: 
						continue
				else:
					break
		if len(midpt_features)>names_len: #don't allow more features than the number of joints
			break '''
	print 'midpoint features; ', midpt_features
	#midpt_features = sim_inds[0:6]

	feature_inds = inter_features + midpt_features
	#print feature_inds
	midpoint_multi = len(inter_features)/(-1.*len(midpt_features)) #makes midpoint features worth as much as interjoint features
	new_features = np.hstack((features[:,inter_features],midpoint_multi*features[:,midpt_features]))


	return new_features, feature_inds
