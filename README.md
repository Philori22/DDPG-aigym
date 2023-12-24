# DDPG: Deep Deterministic Policy Gradient - Pytorch Implementation

## Introduction
A python Implementation of an Agent for Reinforcement learning with Continuous Control using Deep Deterministic Policy Gradient Algorithm  (Lillicrap et al. [arXiv:1509.02971.](https://arxiv.org/abs/1509.02971) ) in pytorch.

The DDPG algorithm uses deep neural networks - two networks an Actor and Critic Network - to approximate the policy and value functions of a Reinforcement Learning problem.


The name DDPG or Deep Deterministic Policy Gradients, refers to how the networks are trained. The value function is trained with normal error and backpropagation, while the Actor entwork is trained with gradients found from the critic network. 


The DDPg algorithm is very useful, can be used for optimal control for agents with compelx configurations, continuous action spaces and high dimensional spaces (e.g. image data too). Some examples of its use are for humanoid robots, drones, cars, or any robotic configuration which may be appropriate.


### Actor Network
The actor network approximates the policy function:<br>
A(s) -> a

where $s$ is a state, and $a$ an action

### Critic Network
The critic netowrk approximated the value function:<br>
C(s,a) -> q

where s represents a state, a an action, and q represents the value of the given state-action pair.

## Examples applied to in this repository:
### AIgm Cheetah

![alt text](https://github.com/Philori22/DDPG-aigym/blob/main/aigym-cheetah.gif)


# Getting Started

## Dependencies
* Pytorch
* OpenAI gym
* Mujoco_py
* Tensorboard
* CUDA (An NVIDA gpu would speed up the process)

# Usage

## Training
- run the following command: python3 ```train.py```
- training will complete once the agent reaches 1.5m timesteps (1500 episodes) in ```train.py```
- after training, the ddpg actor and critic model files will be saved with the trained model weights in ```/models```
- average reward per episode collected and written via tensorboard in ```/runs``` folder, can be downlaoded as csv file

## Test

- run the following command: python ```main.py```
- trained weights model weights will be loaded
- Cheetah aigym environment will be loaded and run, using the trained agent

# TODOs
- [] retrain and test for all AI-gym continuous state space environments
- [] modify ```train.py``` and ```main.py``` to accept parameters to: change environemnts, hyperparameters etc.

# Credits:
* Paper: Continuous Control with deep reinforcement learning by Lillicrap et al. [arXiv:1509.02971.](https://arxiv.org/abs/1509.02971) 
