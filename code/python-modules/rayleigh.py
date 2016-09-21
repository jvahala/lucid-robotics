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
	def __init__(self,task_obj=None,position=0): #initializes the mixed rayleigh based on the current task definition
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
		#print 'transition times/mean values: ', means
		sigmas = getSigmas(means)
		#print 'sigmas; ', sigmas
		#build the rayleigh distribution objects
		#print 'states left: ', states_left
		for k in np.arange(states_left): 
			key = k 
			value = Rayleigh(sigmas[1+k])
			#print 'k/v: ', key,value.sigma
			self.dists[key] = value


	def updateSelf(self,new_position): 
		def getSigmas(means): 	
			#definition of Rayleigh distribution: mean = sigma*sqrt(pi/2) (sigmas are the modes of each rayleigh distribution)
			scaler = np.sqrt(0.5*np.pi)
			sigmas = [x/scaler for x in means]
			return sigmas

		self.scale = sum(self.task.times[new_position:])
		self.dists = {}
		self.position = new_position 

		path = self.task.path
		times = self.task.times
		position = self.position

		states_left = len(path)-position
		means = [0]+[(x+sum(times[position:i]))/float(sum(times[position:]))*self.scale for i,x in enumerate(times) if i>=position]
		#print 'transition times/mean values: ', means
		sigmas = getSigmas(means)
		#print 'sigmas; ', sigmas
		#build the rayleigh distribution objects
		#print 'states left: ', states_left
		for k in np.arange(states_left): 
			key = k 
			value = Rayleigh(sigmas[1+k])
			#print 'k/v: ', key,value.sigma
			self.dists[key] = value

	def proportionate(self,frame_count,proportion_scalar=10.0):
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
		#print 'Trouble with values: ', values 
		if np.amax(values) != 0: 
			prop_values = values/np.amax(values)
			#print 'proportional state values, ',prop_values
			for d in self.dists: 
				for s in proportions: 
					#print 'd,s', d,s,self.position
					if self.task.path[self.position+d] == s: 
						proportions[s] += prop_values[d]
						#print 'proportions[s], s =', s, proportions[s]
						break
			#print 'proportion dict: ', proportions
			#print 'Trouble in proportionate: ', proportions.values()
			totalvalue = np.sum(proportions.values())
			for k in proportions:
				proportions[k] = proportions[k]/totalvalue * proportion_scalar
			return proportions
		else: 
			return -1

def plotMixed(mixed):
	import matplotlib.pyplot as plt
	import seaborn as sns 
	sns.set_context("paper")

	ax =plt.subplots()

	path = mixed.task.path 
	times = mixed.task.times 
	points = [4.1, 26, 41, 57, 79]
	example_point = 13.3
	colors = ['#7ef4cc','#0652ff','k']
	labs = ['Current State','Next','Third','Fourth','Fifth']
	markers = ['s','o','^']
	x = np.linspace(0,mixed.scale,200)

	for i,dist in mixed.dists.iteritems():
		y = dist.pdf(x)
		plt.plot(x,y,color=colors[path[i]])
		if i in [1,2,3]:
			plt.plot([points[i]],[dist.pdf(points[i])],marker=markers[path[i]],color=colors[path[i]],label='State '+str(path[i]))
		else: 
			plt.plot([points[i]],[dist.pdf(points[i])],marker=markers[path[i]],color=colors[path[i]])
		if i == 0: 
			plt.text(points[i]+1.3,dist.pdf(points[i]),labs[i],color=colors[path[i]])
		else:
			plt.text(points[i],dist.pdf(points[i])+0.004,labs[i],color=colors[path[i]])
		if i == 1: 
			plt.annotate('current time', xy=(example_point,dist.pdf(example_point)+0.005), xytext=(example_point+4,dist.pdf(example_point)+0.05),fontsize=12,arrowprops=dict(arrowstyle="simple",fc="0.4", ec="none",connectionstyle="arc3,rad=0.3"))
			plt.plot(example_point,dist.pdf(example_point),color = 'k',marker='*')
		else:
			plt.plot([example_point],dist.pdf(example_point),color = 'k',marker='*')
		plt.legend()

	plt.xlabel('Time Since Last State Change (frames)')
	plt.ylabel('Distribution Value')


	


