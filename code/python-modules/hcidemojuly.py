import main
import test
import process
import utils
import visualize

'''layout of demonstration:

1. previous work 
	- video of a handover with an intermediate task 
	- explanation of approach (labeling of states for both users)
	- application to robot arm (proactive,reactive,adaptive)
	- results (user experience up / efficiency slightly worse than proactive)
2. new work 
	- overall description (define features, spectral clustering to get states and a prime subspace, probabilistic kNN to assign realtime features to states, compute expected time to completion and update robot position along trajectory, update states by projecting new points into the prime subspace for the task)
	- example video 
	- joints video after first handover with path description
	- real time state definition with kNN path description -> path description after state updating (ground truth)
3. next up 
	- probabilistics 
	- robot implementation and study (example collaborative tasks???)

'''

#get all necessary variables for p2-3 video example when loading this module 
receiver,giver,starts,task,curr_labels = main.begin()

#get video of initial task as color-labeled skeleton

def video(name_list,raw_data,labels,fps=15):
	label_labels = ['State 0', 'State 1', 'State 2', '','','']
	fig_num = 1
	title_str = 'Receiver Classification \n First Handover'
	note = 'states generated through spectral clustering'
	visualize.plotMovie3d(name_list,raw_data,labels,label_labels, fig_num, title_str, note, fps)
	#plt.show()

joint_names = receiver.names_list 
raw_data_1 = receiver.data_array[starts[0]:starts[1],:]
video(joint_names, raw_data_1, task.labels)

#print initial task state-transition-path 
print '\n-----------------------------------\nAfter first viewed task: \n-----------------------------------\n'
init_path_info = task.printTaskDef()	#prints the path information in a good way
print 'labels: \n', task.labels 

#go through the next handover and print the generated state-transition-path and labels 
print '\n-----------------------------------\nOnline classification of next task through kNN: \n-----------------------------------\n'
curr_task_num = 1
online_task_labels = main.iterateTask(curr_task_num,receiver,starts,task)
online_path = task.definePath(online_task_labels)
dummy_task = process.Task(receiver)	#create dummy task object to printed out the task definition
dummy_task.path = online_path[0]
dummy_task.times = online_path[1]
online_path_info = dummy_task.printTaskDef()	#prints the online path information in a good way
print 'labels: \n', online_task_labels

#update the states and print the ground truth labels for the online task 
print '\n-----------------------------------\nAfter second task completes and states are updated: \n-----------------------------------\n'
curr_extrema = [starts[1],starts[2]]
task.update(receiver, curr_extrema)
print 'New expected path after update:'
update_path_info = task.printTaskDef()		#new expected task definition after updating 
update_task_labels = task.labels[len(task.history[0]):]
update_path = task.definePath(update_task_labels)
dummy_task.path = update_path[0]
dummy_task.times = update_path[1]
print 'Second task path definition ground truth:'
update_path_info = dummy_task.printTaskDef()
print 'labels: \n', update_task_labels






