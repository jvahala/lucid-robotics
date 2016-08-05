'''
module: kinectData.py 
use: contains functions and class associated with messing with storing and manipulating kinect data  
'''

import numpy as np 
import timeObject
import feature_tools
import utils

class kinectData(object): 

	'''
	Purpose: 
	Manipulate Kinect data files 

	Class functions: 
	addData(self,filename) - Adds Kinect data to the dataArray object as appended rows
	getFeatures(self) - Computes similary matrix for selected features for any data added through addData()

	Required modules: 
	numpy
	timeObject
	feature_tools
	utils
	'''

	def __init__(self, ID): 
		self.ID = ID 					#user ID number (0 is first person, 1 is second, etc)
		self.names_list = []				#list object of the names of the features of data_array
		self.names_base = []			#base names of the name list (ie ShoulderLeft instead of ShoulderLeft.X)
		self.data_array = []			#m by n rows represent different timestamps, columns represent raw data elements from names_list
		self.dataXYZ = np.zeros(1)		#data given as an m by p by 3 where m is number of time stamps, p is number of names_base elements, and 3 is x,y,z
		self.raw_times_array = []		#raw times of each row
		self.num_vectors = 0			#number of timestampes (ie. number of rows)
		self.date = ''					#MM/DD element that is necessary but useless in general
		self.init_time = 0				#initial element's timestamp datetime object
		self.total_time = 0				#total time spanned by the dataset
		self.delta_time_array = []		#datetime.timedelta objects of each row's elapsed time since the start
		self.feat_array = np.zeros(1)	#thresholded feature columns taken from self.all_features - call getFeatures() to fill 
		self.all_features = np.zeros(1)	#feature array for the data containing all features for each frame - call getFeatures() to fill
		self.feature_norms = -1			#normalizing values for all features that are kept at the first getFeatures() call unless this is set back to -1
		self.similarity_method = -1 	#similarity method determines how to generate the similarity matrix in utils.getSimilarityArray()
		self.norm_features = ['SpineMid.X', 'SpineShoulder.X'] #normalize features by the difference between those defined here
		self.norm_value = -1			#value to normalize features by getFeatures()
		self.midpoint = np.zeros((1,3)) #1 by 3 array midpoint(X,Y,Z) between the two parties(u1 and u2) - define in main using feature_tools.getMidpointXYZ(u1_jointsXYZ,u2_jointsXYZ)
		self.feature_inds = -1 			#indicies of the features chosen based on the definition provided in feature_tools.py 
	

	def addData(self, filename): 

		'''
		Purpose: 
		Adds Kinect data to the dataArray object as appended rows

		Inputs: 
		filename: 				Kinect data file [1st row (0 index) is names of variables, additional rows (>0 index) are data]

		Outputs: 
		self.names_list: 		updated if empty
		self.data_array: 		new rows of data from filename are appended 
		self.num_vectors: 		updated to reflect number of data frames added
		self.init_time: 		updated if first set up data added to object
		self.date: 				updated as necessary
		self.total_time:		updated to reflect total time elapsed between init_time and the last timestamp
		self.delta_time_array: 	updated with new delta_time values 
		self.raw_times_array:	updated with newly added raw timestamps

		'''

		#define local variables
		avoid_words = ['timestamp','personID','HandLeftStatus','HandRightStatus']  #define words to not include in names_list
		temp_names_list = []
		data_vector = []
		print 'Adding new data from', filename, '...'

		with open(filename,'r') as f:
			line_index = 0
			
			## begin looking at each line
			for line in f:
				word_index = 0

				## begin looking at each word
				for word in line.split():
					if line_index == 0 and len(self.names_list)==0: 	#only append to names_list if it is empty
						if word not in avoid_words:
							temp_names_list.append(word)
					elif line_index > 0: 
						if word_index == 0 and self.date!=word:
							self.date = word 
						if word_index == 1: 
							if len(self.data_array) == 0:
								self.init_time = timeObject.timeObject(self.date,word)
							curr_time = timeObject.timeObject(self.date,word)
							delta_time = curr_time.time - self.init_time.time
						#if word_index == 2 and word == 'PM': 
							#currTime = armyTimeIt(currTime) 		#implement this function
							#print '' '''continue?'''
						if word_index == 3:
							if word != self.ID: 					#test that the correct person's data is going into the data_array
								break
						if word_index >= 4 and word != 'Tracked' and word != 'Inferred': 
							data_vector.append(float(word))
					word_index += 1 								#increment to next word in line

				# before moving to new line, add data_vector to data_array if data was collected or append names_list if line_index == 0
				if line_index > 0 and len(data_vector) > 0: 
					if len(self.data_array) == 0: #perform if there is nothing yet in the d
						self.data_array = data_vector
					else: 
						data_vector = np.array(data_vector)
						self.data_array = np.vstack((self.data_array,data_vector))
					self.delta_time_array.append(delta_time)
					self.raw_times_array.append(curr_time.time)
				elif line_index == 0 and len(self.names_list) == 0: 
					self.names_list.extend(temp_names_list)
				data_vector = [] 									#reset the data_vector to collect more data
				line_index += 1										#increment to next line
			self.total_time = delta_time

		self.num_vectors = len(self.data_array)
		print ' Data added.\n'
		#end addData()

	def getFeatures(self):

		'''
		Purpose: 
		Builds an array of features using one of the various accepted self.feature_method types
		(ADD in normalization input???? where to normalize the data? - no normalization is implemented)

		Inputs: 
		self.norm_value: 		Not exactly an input, but function will calculate norm value if indef and returned array is normalized by this value
			currently based on |SpineMid.Y - SpineShoulder.Y|
		
		Outputs: 
		self.all_features: 		updated with all feature vectors associated with each frame of kinect data
		self.feat_array: 		updated with the new frames of chosen feature vectors for each frame of kinect data

		'''

		# if no norm value is assigned, set up the norm value
		if self.norm_value == -1: 
			try:  
				name_index_1 = self.names_list.index(self.norm_features[0])
				name_index_2 = self.names_list.index(self.norm_features[1])
				#print 'important: \n', self.data_array[0,name_index_1:name_index_1+3], '\n', self.data_array[0,name_index_2:name_index_2+3]
				#self.norm_value = np.absolute(self.data_array[0,self.names_list.index(self.norm_features[0])] - self.data_array[0,self.names_list.index(self.norm_features[1])])
				self.norm_value = 10.0*np.linalg.norm(self.data_array[0,name_index_1:name_index_1+3] - self.data_array[0,name_index_2:name_index_2+3])**2
				#print self.norm_value
			except ValueError: 
				self.norm_value = 1
				print 'ERROR: norm_features not found.\n'

		#if no features yet defined, start messing with all data
		if self.all_features.shape == (1,): 
			sub_data_array = self.data_array 
		#else if features are defined, mess with only the new data
		else: 
			sub_data_array = self.data_array[(len(self.all_features)-1):self.num_vectors-1,:]

		#define the new feature vectors for each row
		for row in sub_data_array:
			#for each row of data, 
			#	a) get jointsXYZ for row, 
			#	b) get normalized interjoint features, 
			#	c) get normalized joint-midpoint features, 
			#	d) concatenate features together
			self.names_base, jointsXYZ = feature_tools.disectJoints(self.names_list,row)
			features_interjoint = utils.normalize(feature_tools.getInterjointFeatures(jointsXYZ),self.norm_value)
			features_jointMidpoint = utils.normalize(feature_tools.getJointToMidpointFeatures(jointsXYZ,self.midpoint),self.norm_value)
			features = np.hstack((features_interjoint,features_jointMidpoint))
			if self.feature_norms == -1: 
				self.feature_norms = 1.*np.amax(features,axis=0)

			features = features/self.feature_norms #normalize all features within themselves, does this make sense to do just with some generic current max? Future data will be poorly compared to eachother...need some definite 0-1 normalizing factor for all new featuers
			#print 'FEATURES: ', features.shape

			#append feat_vec to self.all_features if it has alread been defined
			if self.all_features.shape != (1,):
				self.all_features = np.vstack((self.all_features,features))
				self.dataXYZ = np.vstack((self.dataXYZ,jointsXYZ[np.newaxis,:,:]))
			else: 
				self.all_features = features
				self.dataXYZ = jointsXYZ[np.newaxis,:,:]

		#remove non time-varying features
		'''will need to implement a method to only calculate the required feature_inds features and to append them properly when adding additional sets of data to the same class object'''
		#self.features_interjoint = self.feat_array[:,0:120]
		#self.features_jointMidpoint = self.feat_array[:,120:]
		if self.feature_inds == -1: 
			self.feat_array, self.feature_inds = feature_tools.thresholdFeatures(self.all_features,self.norm_value)
			feature_tools.describeFeatures(self.feature_inds, len(self.names_base), self.names_base)
		else: 
			self.feat_array = self.all_features[:,self.feature_inds]

		


		#end getFeatures()

