# Hacking Black-Box Reward Functions

## Replicating the Experiments

### Dependencies
All experiments were conducted using OpenAI's gym library. In addition, the continuous state and action experiment builds off of the PyBullet Gymperium implementation of HalfCheetah and OpenAI Spinning Up implementation of DDPG. Both libraries must be installed in order to run this experiment.

### Running Experiments
Each of the three experiments has a corresponding test file (test_grid.py, test_slider.py, test_cheetah.py). These can be run to generate and store results from the experiments. This data can then be plotted using plot.py (note that you will have to comment/uncomment different plot code depending on the experiment and what plots you are hoping to create).

## Introduction
This meta-reinforcement learning project aims to improve the process of providing an agent a concrete problem specification and allowing the agent to learn the best policy for behaving in a given environment. An investigation is performed on how giving an agent access to the reward function for a Markov Decision Process will affect its ability to find an optimal policy in its environment. It builds on works such as “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks” [1] and “Transfer in Variable-Reward Hierarchical Reinforcement Learning” [2]. A variety of environments is used, ranging from one with discrete states and actions to one with continuous states and actions.

## Background
A Markov Decision Process (MDP) stores information about the states, actions, a transition function, and reward function for a given environment. The ultimate goal for an agent in an MDP is to find its optimal policy. The optimal policy would be the policy that maximizes the agents total cumulative reward, or return, after a given episode in the environment. 

In normal reinforcement learning, the agent is learning about a single environment and is attempting to find an optimal policy to traverse that environment. Meta-reinforcement learning introduces the agent to multiple environments that are structurally similar. Existing algorithms are able to train a model that can solve new tasks based on a small number of samples [1]. The agent can improve its ability to find an optimal policy in later environments that it explores because it already knows how to solve a similar problem. There has also been research that is able to perform meta-reinforcement learning on environments that have the same transition functions but variable reward functions [2], which inspired the structure of the grid MDPs in this research. Note that in any Markov Decision Process, the agent must explore its environment and record the rewards that it earns in order to learn information about the reward function. However, in many reinforcement learning problems, the programmer defines the reward function but does not offer that information to the agent.

## Problem Setting
This investigation assumes that the problem involves multiple Markov Decision Processes that are similar in structure. The agent is presented with one MDP at a time. Once an episode in its current MDP is completed, the agent will begin a new episode with a new MDP. The reward function is defined by the programmer in these problems, making it possible to provide this information to the agent. For example, the grid environment used in this study has a set of states and actions that are consistent in each MDP in the problem. The only thing that differs are the rewards associated with each type of space in the grid. The agent does not have direct access to the transition function, as is the convention in reinforcement learning. Additionally, the environment is fully observable. In the grid world specifically, the agent has access to the full map of the environment and its position in the map at all times. The objective of this problem is to maximize the agent’s returns after completing a number of MDPs.

A key component of this investigation is that the agent is given advanced information about its environment by having access to the reward function throughout its learning process. Rewards are always stored in the state of the agent such that the information is easily accessible throughout an episode. The goal of this is to make it so the agent does not have to wait until it explores its environment to gain an understanding of the rewards associated with certain states and actions, which may inform its decisions for what actions to take.

## Experiments
### Hypothesis
It was predicted that the ability to know the reward of moving to a given state before having to explore that state will allow the agent to take actions that will maximize its reward. A result of this advantage would be that an agent with access to the reward function would have higher returns from later tasks than an agent without this information.

### Environment with Discrete States and Actions
#### Environment

![Figure 1](/figures/grid.png "Figure 1")

In this environment, the agent’s state is defined as the coordinates of its position in the grid environment. From any space in the grid, the agent can move right, left, up, and down to an adjacent space in the grid. If the current space in which the agent is located is at the edge of the grid, and the agent aims to move in a direction where there are no longer spaces, the agent will wrap around to the space that is on the opposite edge of the grid (Figure 1). 

#### Approaches
The first baseline approach was a randomized approach, where the action chosen by the agent was randomly chosen from one of the possible choices. Baseline Q-learning algorithms were also implemented for both reinforcement learning and meta-reinforcement learning. The difference between these two approaches is that regular reinforcement learning resets the Q-function after each episode, while meta-reinforcement learning does not. Augmented reinforcement learning and augmented meta-reinforcement learning approaches were also created in order to handle states that included information about the reward function. A mapping of coordinates on the grid to reward values was appended to the end of the existing state, which contained the coordinates of the current location of the agent. This new state was used in these augmented approaches so that the agent can take advantage of this extra information.  

#### Experimental Details
For all approaches, the agent must complete 40 tasks, where each task requires the agent to learn a policy in the grid environment. The grid environment was implemented using the OpenAI Gym library in Python. After the agent finishes an episode (reaches the goal state), its state is reset to its initial state at the top-left corner of the grid, and its reward function is randomly redefined to new values. When the rewards are randomly defined, sometimes the blue spaces have a reward of -1 and the orange ones have a reward of -100, and other times to opposite is true. Reaching the goal state always gives a reward of 10. 

#### Results
Allowing the agent to have access to the reward function had no effect on single-task reinforcement learning but made it possible for the agent to increase its returns during meta-reinforcement learning.
 
<div style="justify-content:space-around;">
    <div style="display: inline-block;">
        <img src="/figures/Q-learning/random.png" alt="figure" width="400"/>
    </div>
</div>
    

The approach that gave the agent a random policy that randomly selected an action every time performed the worst (Figure 2). This is an expected result, since the agent has no strategy to improve its returns.

<div style="justify-content:space-around;">
    <div style="display: inline-block;">
        <img src="/figures/Q-learning/single_task.png" alt="figure" width="400"/>
    </div>
    <div style="display: inline-block;">
        <img src="/figures/Q-learning/single_task_augmented.png" alt="figure" width="400"/>
    </div>
</div>
<!-- <img src="/figures/Q-learning/single_task.png" alt="figure" width="400"/>
<img src="/figures/Q-learning/single_task_augmented.png" alt="figure" width="400"/> -->

The single task, regular reinforcement learning algorithms performed slightly better than the random policy (Figure 3). There is likely no upward trend as the number of completed episodes increases because this algorithm tackles each new task as if it has no prior knowledge about similar tasks. However, there is some learning that occurs during a single episode, hence the improvement in returns in comparison to the random policy. There is no difference between the single task algorithm with hacking the reward function and that without because within a single episode, the reward function information in the state never changes. So, the agent is not able to learn anything from this information. 
  
<img src="/figures/Q-learning/multitask.png" alt="figure" width="400"/>
<img src="/figures/Q-learning/multitask_augmented.png" alt="figure" width="400"/>

The meta-reinforcement learning algorithms performed the best overall. Its returns increase as the agent completes more tasks because the agent can use its prior knowledge about the environment to help create a good policy for newer tasks. The meta-learning approach with access to the reward function follows a similar trend, but with higher returns (Figure 4). This is evidence that allowing the agent to have access to the reward function allows it to deduce information about its environment before taking any actions and develop a better policy.

<img src="/figures/Q-learning/multitask_goals.png" alt="figure" width="400"/>
<img src="/figures/Q-learning/multitask_augmented_goals.png" alt="figure" width="400"/>

<img src="/figures/Q-learning/multitask_state_visits.png" alt="figure" width="400"/>
<img src="/figures/Q-learning/multitask_augmented_state_visits.png" alt="figure" width="400"/>

The two meta-reinforcement learning algorithms differed in their policies during a given task. An analysis was performed where the agent was given 30 tasks and had to take 100 steps in the environment for each task. During these tasks, the multitask approach without access to the rewards was able to reach the goal more often than that with access to the rewards (Figure 5). This is likely due to the agent without hacking abilities not being able to effectively differentiate the two different types of spaces in the grid, as it views the negative rewards from these spaces as equally harmful despite their different values. So, the agent tries to reach the goal state as fast as possible no matter the rewards it accumulates on the way, rather than taking a more strategic route like the augmented meta-learning approach. Figure 6 is a plot of how often the agent visits a given space in the grid. So, the agent with access to the reward function sacrifices a lower number of steps in exchange for a better return.

### Environment with Continuous States and Discrete Actions
#### Environment
The environment used for this experiment is a slider environment created in OpenAI Gym. The state stores the position and velocity of the agent, which can move either left or right. The goal is to reach the target velocity for the given task.

#### Approaches
The same approaches for the previous experiment were used for that of the slider environment. However, the approach for augmented reinforcement learn differs slightly. Here, the state was augmented with queries of the reward function rather than the entire reward function like in the grid world. An oracle approach appends the target velocity to the state to provide insight into the agent's ideal performance. Then, an approach with one query and that of two queries was created for testing how accessing the reward function affects agent performance. The agent is able to query the reward function by offering a state s, an action a, and the next state after taking action a in state s. In return, it receives the reward from that state and appends it to the agent's state.

#### Experimental Details
The slider environment was created using OpenAI Gym. In this experiment, the agent completes 600 episodes with a maximum of 25 steps in each. Each episode is a task with a random target velocity between -10 and 10. Each approach was run with 25 seeds and those results were averaged together. 

#### Results
Similar to the grid world experiment, the single task approaches did not show any improvement as the number of episodes increased, but the multitask approaches did.

<img src="/figures/DQN/_eval_single_dqn.png" alt="figure" width="400"/>
<img src="/figures/DQN/_eval_single_augmented_dqn.png" alt="figure" width="400"/>
<img src="/figures/DQN/_eval_multi_dqn.png" alt="figure" width="400"/>
<img src="/figures/DQN/_eval_MultiTaskAugmentedOracle.png" alt="figure" width="400"/> 


Regular meta-reinforcement learning achieved a maximum return of approximately -1000. The oracle approach, however, achieved returns very close to zero. This is likely because the agent knows exactly the target velocity it is trying to achieve by accessing it in its state. 

<img src="/figures/DQN/_eval_multi_augmented_dqn_1_query.png" alt="figure" width="400"/> 

The approach with only one query of the reward function reached returns similar to the regular meta-reinforcement approach. Since the reward function in this environment is only dependent on the current state, it is reasonable that knowing the reward based on a current position and velocity is not enough information to deduce the direction towards the target velocity.

<img src="/figures/DQN/_eval_multi_augmented_dqn_2_query.png" alt="figure" width="400"/> 

The two-query approach, on the other hand, reaches a similar reward close to zero to the oracle approach. This approach provides information about a complete step, so the agent is able to deduce the direction it needs to go to reach the target velocity. For example, a query with a given velocity and a query with a velocity higher than the previous will show whether increasing the velocity improves or decreases the reward. If it improves, the agent knows to continue increasing velocity until it reaches the target. If it decreases, the agent can do the opposite. With this strategy, the agent can optimize its actions and maximize rewards. 

### Environment with Continuous States and Discrete Actions
#### Environment
This experiment uses the popular meta-reinforcement learning environment, HalfCheetah. The state keeps track of the positions and velocities of different parts of the cheetah. The actions are NO ONE KNOWS LOL. The goal in this environment is to make the cheetah move as far as possible in the target direction.

#### Approaches
The approaches for this experiment are similar to the slider experiment. However, a single task approach was not tested because it is known that a single task approach can not adapt to a new task in such a complex environment. In addition, the two query approach is not included because one query is sufficient for deducing the target direction. An automatic query approach is added in order to test the agents ability to access the reward function more independently. Instead of hard-coding the inputs into the query for the reward function, the agent uses its first observed transition in the environment as inputs for the query.

#### Experimental Details
The PyBullet Gymperium implementation of HalfCheetah and OpenAI Spinning Up implementation of DDPG were used for this experiment. In order to produce comparable results to the Spinning Up DDPG experiments with MuJoCo HalfCheetah, the same hyperparamaters from their experiments were used for this one. The agent completes 200 episodes in the environment, each with 1000 steps. Each episode is a task that requires the cheetah to move in a random direction, either right or left, to earn rewards. Every tenth episode is used for evaluation, which in this case means not adding any noise to the selected action in a given state. Each approach was tested with 5 different seeds.

#### Results
Providing the agent with access to the reward function again improved its returns in the HalfCheetah environment. 

<img src="/figures/DDPG/MultiTaskDDPG.png" alt="figure" width="400"/> 

The regular meta-reinforcement learning approach was not able to improve its returns over time. This may be due to the agent not being able to detect the changing target direction and finding it optimal to try to stay in place.

<img src="/figures/DDPG/MultiTaskDDPGAugmentedOracle.png" alt="figure" width="400"/> 

This approach improved its returns as the agent completed more tasks. It was able to exceed a return of 6000 (after averaging across all 5 tests with different seeds). This makes sense since the agent has access to the target direction in its state.

<img src="/figures/DDPG/MultiTaskDDPGQuery.png" alt="figure" width="400"/> 

Similar to the previous approach, the agent improves its returns and is able to exceed a reward of 5000. The agent has enough information to deduce the target direction for a given task, so it is expected that it performs similarly to the oracle approach.

<img src="/figures/DDPG/MultiTaskDDPGAutoQuery.png" alt="figure" width="400"/> 

This approach achieves a maximum return of about 4000. This approach is expected to perform similarly to the query approach, the only difference being that it has to observe one transition before it is able to acquire the appropriate inputs to query the reward function. It is possible that this delay is reason for the slight difference in returns. However, it still received returns in a similar range to the query approach. 

## Conclusion

In all cases, the agent was able to improve its performance by having access to the reward function. Future investigations could expand on the automatic query process. The current process may not work in environments with more complex reward functions, such as those with sparse rewards. A more robust query process that relies less on help from a human would help make more complete this strategy of reinforcement learning.

### References
[1] Finn, Chelsea, Abbeel Pieter, and Levine, Sergey. Model-Agnostic Meta-Learning for Fast 
Adaptation of Deep Networks. In International Conference on Machine Learning (ICML), 2017.

[2] Mehta, Neville, Natarajan, Sriraam, Tadepalli, Prasad, and Fern, Alan. Transfer in Variable-
Reward Hierarchical Reinforcement Learning. In Machine Learning, 2008.


Research is conducted by Shariqah Hossain and Tom Silver.
