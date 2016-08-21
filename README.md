# lucid-robotics

Automation of Collaborative Human-Robot Task Modelling 

Prior research has shown that adapting a robot's motion in human-robot collaborative tasks leads to a better user experience without drastically effecting task efficiency compared to a proactive "speed-based" robot model in which the robot accomplishes its process without concern for user task progress. However, the same research required a deep understanding of the task itself to build the model, and thus it was an intensive process for additional collaborative models to be produced. 

This new work addresses these issues by automating the process of collaborative task modelling to allow for the online creation and adaptation to unseen tasks so that a robot partner may work in an optimal manner with its human partner. 

This algorithm relies upon spectral clustering and dynamic time warping for developing new task models after full task processes have been seen, and it relies on a probabilistic k nearest-neighbors (kNN) method to determine current task progress for the robot to alter its movement online. 

The algorithm can be demonstrated in the robotic simulation software VREP by:
0. Make sure the file path in the Head childscript in mico_simulation_main.ttt is set to a kinect data file path.
1. Run pyvrep.py, then starting the scene mico_simulation_main.ttt in vrep.
2. When ready, press the PLAY button in the vrep scene to start motion.

Colors in the demonstration will be related to either the current state or the current task depending on the current setting found in the sceneUI, once that new ui is actually implemented. Until then, the colors represent the current state. 
