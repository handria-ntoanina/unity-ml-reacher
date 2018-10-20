# unity-ml-reacher
This repository contains an implementation of reinforcement learning based on DDPG but using parallel agents to solve the unity reacher environment. It has a 20 double-jointed arms. Each one has to reach a target. Whenever one arm reaches its target, a reward of up to +0.1 is received. This environment is simliar to the [reacher of Unity](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).<br/>
The action space is continuous [-1.0, +1.0] and consists of 4 values for 4 torques to be applied to the two joints. <br/>
`The environment is considered as solved if the average score of the 20 agents is +30 for 100 consecutive episodes.`<br/>
A video of a trained agent can be found by clicking on the image here below <br/>
[![Video](https://img.youtube.com/vi/6s2ejba1s_s/0.jpg)](https://www.youtube.com/watch?v=6s2ejba1s_s)
## Content of this repository
* __report.pdf__: a document that describes the details of  implementation, along with ideas for future work
* folder __agents__: contains the implementation of
	* a parallel DDPG using one network shared by all agents
	* a parallel DDPG with multiple network
	* an Actor-Critic network model using tanh as activation
	* a ReplayBuffer
	* an ActionNoise that disturb the output of the actor network to promote exploration
	* a ParameterNoise that disturb the weight of the actor network to promote exploration
	* an Ornstein-Uhlenbeck noise generator
* folder __started_to_converge__: weights of a network that started to converge but slowly
* folder __weights__: weights of the network that solved this environment. It contains as well the history of the weights.
* jupyter notebook __Continuous_Control.ipynb__: run this notebook to train the agents
* jupyter notebook __noise.ipynb__: use this notebook to optimize the hyperparameter of the noise generator to check that its output would not limit the exploration
* jupyter notebook __view.ipynb__: a notebook that can load the different saved network weights and visualize the agents
## Requirements
To run the codes, follow the next steps:
* Create a new environment:
	* __Linux__ or __Mac__: 
	```bash
	conda create --name ddpg python=3.6
	source activate ddpg
	```
	* __Windows__: 
	```bash
	conda create --name ddpg python=3.6 
	activate ddpg
	```
* Perform a minimal install of OpenAI gym
	* If using __Windows__, 
		* download swig for windows and add it the PATH of windows
		* install Microsoft Visual C++ Build Tools
	* then run these commands
	```bash
	pip install gym
	pip install gym[classic_control]
	pip install gym[box2d]
	```
* Install the dependencies under the folder python/
```bash
	cd python
	pip install .
```
* Create an IPython kernel for the `ddpg` environment
```bash
	python -m ipykernel install --user --name ddpg --display-name "ddpg"
```
* Download the Unity Environment (thanks to Udacity) which matches your operating system
	* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
	* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
	* [Windows (32-bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
	* [Windows (64 bits)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

* Start jupyter notebook from the root of this python codes
```bash
jupyter notebook
```
* Once started, change the kernel through the menu `Kernel`>`Change kernel`>`ddpg`
* If necessary, inside the ipynb files, change the path to the unity environment appropriately

