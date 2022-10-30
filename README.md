# Introduction to OpenAI Gym

This document serves as an introduction to the implementation logic of reinforcement learning(RL) with [Gym](https://www.gymlibrary.dev/) and the [multi-agent particle environment](https://github.com/openai/multiagent-particle-envs). I do not discuss the details of coding such as how action spaces are defined in Gym.

To avoid possible display problems of math expressions on the repository page, please read in the [preview](/README.md). 

## Installation and Introduction

For Gym installation, I recommend to start a new Anaconda environment, and build the Gym environment for your project inside it:

```bash
conda create -n 'CondaEnvName'
conda activate 'CondaEnvName'
pip install gym==0.10.5
```

This command installs core Gym APIs with basic Gym environments such as [`Atari`](https://www.gymlibrary.dev/environments/atari/). If you want to install gym with additional environments such as [`MuJoCo`](https://www.gymlibrary.dev/environments/mujoco/), use

```bash
pip install gym[mujoco] 
```

Taking the MuJoCo environment as an example, a Gym environment usually contains the following contents in logic: 

![Untitled](document.assets/Untitled.jpeg)

- ***World physics*** describe the rules based on which the environment works. For example, supported by the [DeepMind MuJoCo engine](https://github.com/deepmind/mujoco), the MuJoCo enviroment have physics laws such as gravity, collision, joint constraints… In constrast, the atari environment has no gravity but has rules such as how bullets fly.
- ***Environments*** are built based on the world physics and provide an environment for RL. For example, Ant_v4 puts an ant on a plane and users can teach the ant to walk with RL methods inside this environment. Different ***scenarios*** can be set if we hope the ant to learn different skills in this environment.

Next, we' ll talk about how Gym is used in reinforcement learnings.

## RL in Gym environments

In reinforcement learning, the following elements are required:

- ***Agent*** obtain information $o_t$ and chooses certain actions from its policy at each time step: $p(a_t\mid s_t) = \pi (o_t)$
- ***Environment*** is influenced by the agents' actions and changes its states based on physics laws: $s_{t+1} \sim \mathcal{P} (s_{t+1} \mid s_t,a_t)$
- ***Reward*** is fed to the agents from the environment to enable them learn from their experiences: $r_t = f_r (s_t,a_t)$

The role of Gym environment in RL is right as its name 'environment'  — get action $a_t$ from agents, step to another state $s_{t+1}$ basaed on world rules, and feedback rewards $r_t$ — illustrated as the picture below:

![Untitled](document.assets/Untitled%201.jpeg)

And following is a commonly seen call to the Gym enviornment APIs in RL training process (*multi-agent centralized* *training* as an example):

```python
import gym
env = gym.make('CollectiveManipulation_Env-v0')

import Agents
agents = Agents.init(num = 50)

episodes = 10000
for episode in range(episodes):
	accumulated_reward = 0
	env.reset()
	for time in range(300):
		actions = []
		obs = env.get_obs()
		for i in range(agents):
			action = agents[i].select_action(obs[i])
			actions.append(action)
		state, reward = env.step(actions)
		env.render()
		accumulated_reward += reward
	agents.policy_update(reward)
env.close()
```

The lines below show core functions of a Gym environment and how Gym environments work in RL: 

```python
# start an environment
env = gym.make('CollectiveManipulation_Env-v0')
# initialze the environment before episode starts
env.reset()
# get observation from its current state
obs = env.get_obs()
# step to another state based on its rules, and return a reward
state, reward = env.step(actions)
# render the GUI if possible
env.render()
# close the environment
env.close()
```

And these functions are important for us to build our own environments.

## Build a Gym environment

> Notice: This document is written to describe logic. for a detailed descripiton, please refer to the official [gym documents](https://www.gymlibrary.dev/content/environment_creation/).
> 

A Gym environment is a Python class which holds the above core functions, and it is usually defined similar to the following form:

```python
import gym

class CollectiveManipulationEnv(gym.Env):
	def __init__(self,world, actors, init_state):
		self.world = world
		self.actors = actors
		self.state = init_state
	
	def get_obs(self):
		return obs
	
	def step(self, actions):
		self.state = rules(actors, actions, world)
		reward = self._get_reward(self.state)
		return state, reward

	def render(self):
		# render environment
	
	def reset(self):
		self.state = init_state

	def _get_reward(self, state):
		return reward
```

Only defining a class will not enable us call the Gym APIs, it must be *registered* and *wrapped into a Python package* according to corrent file structures. View the [official document](https://www.gymlibrary.dev/content/environment_creation/) for details. And finally, the package needs to be installed into your Anaconda environment:

```bash
cd PackageFolder
pip install -e PackageName
```

# Multi Agent Partical Environment

> This is a simple environment but with relative **comprehensive physics world properties** such as communication, collision, force and so on. We can use it to run our RL with some modification. It is **light-weight** compared to other environments, and its good for our large-scale settings.

View their [github repository](https://github.com/openai/multiagent-particle-envs).

## Introduction

This Gym environment package is structured as the picture below.

- `core.py` In 'core.py', the basic properties of world and entities (agents and obstacles) are defined, also their states, actions, and physics interaction laws.
- `environment.py` Based on 'core.py', the basic functions such as `step()`, `render()`, and `reset()` are defined. The action space and observation space are also fixed here in `__init__()`.
- ***scenarios***: different scenarios have different agents and rewards. They are coded for different tasks. For example, in `simple.py` there is only one agent, one goal and no communication. In `simple_reference.py` there are multiple agents and communication between them.

![image-20221027111433069](README.assets/image-20221027111433069.png)

To run this environment, a sample code is as follows:

```python
import multiagent
from make_env import make_env

def main(scenario_name):
    env = make_env(scenario_name)
    env.reset()
    for i in range(300):
        env.step([[0,1,1,1,1]]) # notice the action space
        env.render()
    env.close()

main('simple')
```

We need to use `make_env()` function and input the file name of the desired scenario. The rest parts are the same as other gym environments. Notice that this environment only has 1 channel (5 dimension) for physical action input and 1 channel for communication action input. (See the `_set_action()` in environment.py). So we need to modify its `core.py` and `environment.py`. Then design our own scenarios for different simulation purposes.

For more comprehensive understanding, read the following content and check `core.py`, `environment.py`, `simple_environment.py` to figure out how worlds are set, actions are delivered into the world, and how the worlds are updated.

## World Physics

> Describe the entity states(properties), interaction rules and action inputs to the current particle environment.

In `core.py`, the relationships between classes (green) are illustrated in the picture below. `Entity` is the parent class of `Landmark` and `Agent`, which contains `action` and `AgentState`. The `World` class contains all possible entities, and describes how forces are added to propel their motions.

![image-20221029011303597](README.assets/image-20221029011303597.png)

### Entities and States

- ***Entities*** have properties such as color, mass, max speed and a constant acceleration. Users can also define if the entity is movable or collision enabled.
- ***Agents*** have other properties like if communication allowed, noise, and control input range. Notice that the agent here is different from what it is meant in RL. As a part of the environment, it has no policy and can only execute an input action $a_t$, stepping to another state $s_t$. Thus, I do not recommend define learning policy/network in this `Agent` class.
- ***AgentState*** includes its position, velocity and current communication signal to exchange information with other agnets.

### Interaction Rules (Dynamics)

In `World` class, how forces are applied are described. For an agent $i$, its dynamics is governed by

$$\begin{equation} m_i \frac{d\boldsymbol{v}^i}{dt}={\boldsymbol{u}}_i + \sum_{j\in \mathcal{R}(i)} A \log\left[1+\exp(-\frac{r_{ij}-d_m}{B})\right]\cdot {\boldsymbol{p}}_{ij} \end{equation}$$

where a repulsion force is generated between entities if the distance is smaller than the minimum allowed distance. $\boldsymbol{u}_i$ is derived from the action input as a self-driving force of an agent. And the second part maps the distance of its neighboring entities to repulsive forces.

### Action Input

In `environment.py`, the agents' action space is divided into the force space `u_action_space` and communication space `c_action_space`. The action input is either discrete numbers (0,1…N) or an one-hot N dimensional vector, depending on boolean `discrete_action_input`. And the input are mapped to accleration on different discrete dirctions and communication with different agents. And the acceleration generates the motion of agents further. For more details, refer to `_set_action()`.

![image-20221029011319601](README.assets/image-20221029011319601.png)

## Modification

> For the simulation setup of different multi-agent researches, the environment should be modified.

To modify this environment, one should first figure out his/her own RL model, i.e. a Markovian decision process $\mathcal{M}=<\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{O},\mathcal{R},>$, where the state transition probability matrix $\mathcal{P}: \mathcal{S}\times \mathcal{S} \mapsto \left[0,1\right]$ is decided by the world dynamics. Then based on your RL model, change `core.py`, `environment.py` respectively and define new scenarios.

Take the world dynamics of RL in our research as an example:

In our research, the ***state space*** ( $\mathcal{S}$ ) contains a vector $\boldsymbol{h}$, which is the intention (desired velocity) of an agent. And the motion agents are controlled by the $\boldsymbol{h}$ as an input. Dynamics ( $\mathcal{P}$ ) of agents are governed by the *social force model*:

$$\begin{equation} m_i \frac{d\boldsymbol{v}^i}{dt} = m_i \frac{\boldsymbol{h}^i - \boldsymbol{v}^i}{\tau} + \sum_{j\in \mathcal{R}(i)} f_{ij} + f_{ik}\end{equation}$$

where $f$ denotes the repulsive forces between entities (neighboring agent $j$ and obstacle $k$). Therefore, in `core.py`, we need to add $\boldsymbol{h}$ to both `AgentState` and `Action`. And slightly change the codes in `get_collision_force()` to make them consistent with our dynamics.

The ***action space*** ( $\mathcal{A}$ ) for one of our RL problems is a 3-dimensional discrete vector space, so in `environment.py`, the `action_space` property of class `MultiAgentEnv`.

In our simulation, some agents can learn to control the intention $\boldsymbol{h}$ of other agents based on their observation ( $\mathcal{O}$ ) of neighbors' behavior and a task oriented global reward ( $\mathcal{R}$ ). These are described by creating a new scenario, and implementing our own `observation` and `reward` functions. Notice that other than the two functions mentioned, new scenarios needs two other functions `make_world()` and `reset()` to help create the world and initialize.
