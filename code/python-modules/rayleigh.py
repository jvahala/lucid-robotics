#from scipy.stats import rayleigh
#import matplotlib.pyplot as pyplot
import numpy as np

class Rayleigh(object): 
	def __init__(self,sigma): 		#sigm value represents the mode of the distribution
		self.sigma = float(sigma)
	
	def pdf(self,x):
		def calculate(x): 
			value = (x/(self.sigma**2))*np.exp((-x**2)/(2*self.sigma**2))
			return value
		try: 
			if x>=0: 
				return calculate(x)
			else: 
				return 'ERROR: Rayleigh distribution is only defined at x >= 0.'
		except ValueError,TypeError:  
			y = [calculate(i) for i in x]
			return np.array(y) 
		

class MixedRayleigh(object): 
	'''
	want an object that can be given the task definition and the current location within the task (if the state has 7 transitions, then it should be given for instance that it is in the 3rd state of the list)
	From the task, it can generate the proper distribution variances based on the task.path and task.times. It treats the position input as the highest probability (lowest variance) distribution of the set. 
	There must be some scale set for the entire process, maybe 0 to 10. The increments are based upon the expected task length (sum(task.times)). The crossover points are at the relative time position throughout the task. 

	'''
	def __init__(self,task_obj,position=0): #initializes the mixed rayleigh based on the current task definition
		def getSigmas(means): 	
			#definition of Rayleigh distribution: mean = sigma*sqrt(pi/2) (sigmas are the modes of each rayleigh distribution)
			scaler = np.sqrt(0.5*np.pi)
			sigmas = [x/scaler for x in means]
			return sigmas

		#setup base variables 
		self.scale = sum(task_obj.times[position:])		#essentially how the expected times get thrown onto - ie this corresponds to the expected time of a full task if the position is 0
		self.dists = {}			#dictionary of the Rayleigh objects sorted by position in the state-transition-path
		self.task = task_obj 
		self.position = position
		#from the task_obj, collect the number of states remaining from given position
		states_left = len(task_obj.path)-position 
		#convert the expected state times into positions on the default scale, 0 -> 0, 10 -> 10/sum(times[position:])*self.scale, 5 -> 0+10+5 / sum(times[position:]) * self.scale
		means = [0]+[(x+sum(task_obj.times[position:i]))/float(sum(task_obj.times[position:]))*self.scale for i,x in enumerate(task_obj.times) if i>=position]
		print 'transition times/mean values: ', means
		sigmas = getSigmas(means)
		print 'sigmas; ', sigmas
		#build the rayleigh distribution objects
		print 'states left: ', states_left
		for k in np.arange(states_left): 
			key = k 
			value = Rayleigh(sigmas[1+k])
			print 'k/v: ', key,value.sigma
			self.dists[key] = value


		#define variances for each transition time to be me

	def proportionate(self,frame_count):
		'''
		Purpose: 
		Returns the proportions of each distribution at the input location in the mixed Rayleigh distribution soas to scale the counts returned from kNN. 

		Inputs: 
		frame_count - frames since last transiton point occurred. For instance, if it was determined that we had moved into the second state of the state-transition-path (state 1), and that there had been 17 frames since that transition occured, then frame_count = 17. 

		Outputs: 
		proportions - ndarray object of the relative proportions of each state
		'''
		num_states = self.task.num_states 
		proportions = {} #will fill with each possible state as a key and proportions as the values
		if frame_count == 0: 
			for s in np.arange(num_states): 
				proportions[s] = 1
			return proportions 
		for s in np.arange(num_states): 
			proportions[s] = 0
		values = np.array([self.dists[x].pdf(frame_count) for x in self.dists])
		prop_values = values/np.amax(values)
		#print 'proportional state values, ',prop_values
		for d in self.dists: 
			for s in proportions: 
				#print 'd,s', d,s
				if self.task.path[self.position+d] == s: 
					proportions[s] += prop_values[d]
					#print 'proportions[s], s =', s, proportions[s]
					break
		#print 'proportion dict: ', proportions
		return proportions



