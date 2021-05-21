# Hacking Black-Box Reward Functions

##Replicating the Experiments

###Dependencies

All experiments were conducted using OpenAI's gym library. In addition, he continuous state and action experiement builds off of the pybullet implementation of HalfCheetah and spinning up implementation of DDPG. Both libraries must be installed in order to run this experiment.

###Grid World Experiment

###Slider Experiment

###HalfCheetah Experiment

##Introduction

This meta-reinforcement learning project aims to improve the process of providing an agent a concrete problem specification and allowing the agent to learn the best policy for behaving in a given environment. An investigation is performed on how giving an agent access to the reward function for a Markov Decision Process will affect its ability to find an optimal policy in its environment. It builds on works such as “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks” [1] and “Transfer in Variable-Reward Hierarchical Reinforcement Learning” [2]. A variety of environments is used, ranging from one with discrete states and actions to one with continuous states and actions.

##Background

A Markov Decision Process (MDP) stores information about the states, actions, a transition function, and reward function for a given environment. The ultimate goal for an agent in an MDP is to find its optimal policy. The optimal policy would be the policy that maximizes the agents total cumulative reward, or return, after a given episode in the environment. 

In normal reinforcement learning, the agent is learning about a single environment and is attempting to find an optimal policy to traverse that environment. Meta-reinforcement learning introduces the agent to multiple environments that are structurally similar. Existing algorithms are able to train a model that can solve new tasks based on a small number of samples [1]. The agent can improve its ability to find an optimal policy in later environments that it explores because it already knows how to solve a similar problem. There has also been research that is able to perform meta-reinforcement learning on environments that have the same transition functions but variable reward functions [2], which inspired the structure of the grid MDPs in this research. Note that in any Markov Decision Process, the agent must explore its environment and record the rewards that it earns in order to learn information about the reward function. However, in many reinforcement learning problems, the programmer defines the reward function but does not offer that information to the agent.

##Problem Setting

This investigation assumes that the problem involves multiple Markov Decision Processes that are similar in structure. The agent is presented with one MDP at a time. Once an episode in its current MDP is completed, the agent will begin a new episode with a new MDP. The reward function is defined by the programmer in these problems, making it possible to provide this information to the agent. For example, the grid environment used in this study has a set of states and actions that are consistent in each MDP in the problem. The only thing that differs are the rewards associated with each type of space in the grid. The agent does not have direct access to the transition function, as is the convention in reinforcement learning. Additionally, the environment is fully observable. In the grid world specifically, the agent has access to the full map of the environment and its position in the map at all times. The objective of this problem is to maximize the agent’s returns after completing a number of MDPs.

A key component of this investigation is that the agent is given advanced information about its environment by having access to the reward function throughout its learning process. Rewards are always stored in the state of the agent such that the information is easily accessible throughout an episode. The goal of this is to make it so the agent does not have to wait until it explores its environment to gain an understanding of the rewards associated with certain states and actions, which may inform its decisions for what actions to take.

##Experiments

###Hypothesis
It was predicted that the ability to know the reward of moving to a given state before having to explore that state will allow the agent to take actions that will maximize its reward. A result of this advantage would be that an agent with access to the reward function would have higher returns from later tasks than an agent without this information.

###Environment with Discrete States and Actions
####Environment
In this environment, the agent’s state is defined as the coordinates of its position in the grid environment. From any space in the grid, the agent can move right, left, up, and down to an adjacent space in the grid. If the current space in which the agent is located is at the edge of the grid, and the agent aims to move in a direction where there are no longer spaces, the agent will wrap around to the space that is on the opposite edge of the grid (Figure 2). 

####Approaches
The first baseline approach was a randomized approach, where the action chosen by the agent was randomly chosen from one of the possible choices. Baseline Q-learning algorithms were also implemented for both reinforcement learning and meta-reinforcement learning. The difference between these two approaches is that regular reinforcement learning resets the Q-function after each episode, while meta-reinforcement learning does not. Augmented reinforcement learning and augmented meta-reinforcement learning approaches were also created in order to handle states that included information about the reward function. A mapping of coordinates on the grid to reward values was appended to the end of the existing state, which contained the coordinates of the current location of the agent. This new state was used in these augmented approaches so that the agent can take advantage of this extra information.  

####Experimental Details
For all approaches, the agent must complete 40 tasks, where each task requires the agent to learn a policy in the grid environment. The grid environment was implemented using the OpenAI Gym library in Python. After the agent finishes an episode (reaches the goal state), its state is reset to its initial state at the top-left corner of the grid, and its reward function is randomly redefined to new values. When the rewards are randomly defined, sometimes the blue spaces have a reward of -1 and the orange ones have a reward of -100, and other times to opposite is true. Reaching the goal state always gives a reward of 10. 

##Results
Allowing the agent to have access to the reward function had no effect on single-task reinforcement learning but made it possible for the agent to increase its returns during meta-reinforcement learning.
 
Figure 3

The approach that gave the agent a random policy that randomly selected an action every time performed the worst (Figure 3). This is an expected result, since the agent has no strategy to improve its returns.
  
Figure 4

The single task, regular reinforcement learning algorithms performed slightly better than the random policy (Figure 4). There is likely no upward trend as the number of completed episodes increases because this algorithm tackles each new task as if it has no prior knowledge about similar tasks. However, there is some learning that occurs during a single episode, hence the improvement in returns in comparison to the random policy. There is no difference between the single task algorithm with hacking the reward function and that without because within a single episode, the reward function information in the state never changes. So, the agent is not able to learn anything from this information. 
  
Figure 5

The meta-reinforcement learning algorithms performed the best overall. Its returns increase as the agent completes more tasks because the agent can use its prior knowledge about the environment to help create a good policy for newer tasks. The meta-learning approach with access to the reward function follows a similar trend, but with higher returns (Figure 5). This is evidence that allowing the agent to have access to the reward function allows it to deduce information about its environment before taking any actions and develop a better policy.
  
Figure 6
  
Figure 7

The two meta-reinforcement learning algorithms differed in their policies during a given task. An analysis was performed where the agent was given 30 tasks and had to take 100 steps in the environment for each task. During these tasks, the multitask approach without access to the rewards was able to reach the goal more often than that with access to the rewards (Figure 6). This is likely due to the agent without hacking abilities not being able to effectively differentiate the two different types of spaces in the grid, as it views the negative rewards from these spaces as equally harmful despite their different values. So, the agent tries to reach the goal state as fast as possible no matter the rewards it accumulates on the way, rather than taking a more strategic route like the augmented meta-learning approach. Figure 7 is a plot of how often the agent visits a given space in the grid. So, the agent with access to the reward function sacrifices a lower number of steps in exchange for a better return.

###References
[1] Finn, Chelsea, Abbeel Pieter, and Levine, Sergey. Model-Agnostic Meta-Learning for Fast 
Adaptation of Deep Networks. In International Conference on Machine Learning (ICML), 2017.

[2] Mehta, Neville, Natarajan, Sriraam, Tadepalli, Prasad, and Fern, Alan. Transfer in Variable-
Reward Hierarchical Reinforcement Learning. In Machine Learning, 2008.


Research is conducted by Shariqah Hossain and Tom Silver.
