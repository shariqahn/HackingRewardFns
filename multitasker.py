import gym
import gym_foo
import itertools
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
# import ipdb


def test_single_approach(task_generator, approach, total_num_tasks=30):
    results = []
    task = task_generator('foo-v0')
    approach = approach(task.get_actions(), task.get_rewards)

    for task_num in range(total_num_tasks):
        result = run_approach_on_task(approach, task)
        results.append(result)

    return results

def run_approach_on_task(approach, task, max_num_steps=100):
    # increase num steps to ~1000
    # Task is a Gym env
    state = task.reset()
    # Initialize the approach
    approach.reset()
    # Result is a list of all rewards seen
    result = []
    for t in range(max_num_steps):
        action = approach.get_action(state)
        next_state, reward, done, _ = task.step(action)
        result.append(reward)
        # Tell the approach about the transition (for learning)
        approach.observe(state, action, next_state, reward, done)
        state = next_state
        if done:
            # break
            state = task.reset()
            # reset environment without rerandomizing
            # allows agent to learn for longer
    return result

class TaskGenerator:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def reset(self):
        self.env.randomize_rewards()
        state = self.env.reset()
        return state

    def step(self, action):
        return self.env.step(action)

    def get_actions(self):
        return self.env.action_space

    def get_rewards(self):
        return self.env.reward_function

import abc

class Approach:
        
    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError('Override me!')

    @abc.abstractmethod
    def get_action(self, state):
        raise NotImplementedError('Override me!')

    @abc.abstractmethod
    def observe(self, state, action, next_state, reward, done):
        raise NotImplementedError('Override me!')


from random import randint

class RandomPolicyApproach(Approach):

    def __init__(self, action_space, reward_function):
        self.action_space = action_space

    def reset(self):
        pass

    def observe(self, state, action, next_state, reward, done):
        pass

    def get_action(self, state):
        # return self.action_space.sample()
        index = randint(0,3) # hacky! fix action space
        return self.action_space[index]


class SingleTaskQLearningApproach(Approach):

    def __init__(self, action_space, reward_function, eps=0.1, discount_factor=0.9, alpha=0.5):
        self.action_space = action_space
        self.eps = eps # probability of random action
        self.discount_factor = discount_factor # gamma
        self.alpha = alpha # learning rate
        self.reward_function = reward_function

    def reset(self):
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))\

    def get_action(self, state):
        # Epsilon-greedy: take a random action with probability eps
        if np.random.random() < self.eps:
            # return self.action_space.sample()
            index = randint(0,3) # hacky! fix action space
            return self.action_space[index]
        # Find action with max return in given state
        return self.action_space[np.argmax(self.Q[state])] 

    def observe(self, state, action, next_state, reward, done):
        best_next_action = np.argmax(self.Q[next_state]) 
        td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
        action_index = self.action_space.index(action)
        td_delta = td_target - self.Q[state][action_index]
        # Update the Q function
        self.Q[state][action_index] += self.alpha * td_delta


class MultiTaskQLearningApproach(SingleTaskQLearningApproach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))

    def reset(self):
        # print(self.Q)
        pass

class SingleTaskAugmentedQLearningApproach(SingleTaskQLearningApproach):
    def get_action(self, state):
        # Epsilon-greedy: take a random action with probability eps
        if np.random.random() < self.eps:
            # return self.action_space.sample()
            index = randint(0,3) # hacky! fix action space
            return self.action_space[index]
        # Find action with max return in given state
        return self.action_space[np.argmax(self.Q[self.augment_state(state)])] 

    def observe(self, state, action, next_state, reward, done):
        augmented_next_state = self.augment_state(next_state)
        augmented_state = self.augment_state(state)

        best_next_action = np.argmax(self.Q[augmented_next_state]) 
        td_target = reward + self.discount_factor * self.Q[augmented_next_state][best_next_action]
        action_index = self.action_space.index(action)
        td_delta = td_target - self.Q[augmented_state][action_index]
        # Update the Q function
        self.Q[augmented_state][action_index] += self.alpha * td_delta

    def augment_state(self, state):
        new_state = [int(state[1]), int(state[4])] + sorted(self.reward_function().items())
        return str(new_state)


class MultiTaskAugmentedQLearningApproach(SingleTaskAugmentedQLearningApproach):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Q = defaultdict(lambda: np.zeros(len(self.action_space)))

    def reset(self):
        # print(self.Q)
        pass

if __name__ == "__main__":
    for approach in ('Random Policy', 'Single Task', 'Multitask', 'Single Task with Hacking', 'Multitask with Hacking'):
        if approach == 'Random Policy':
            approach_fn = RandomPolicyApproach
            file = 'random.png'
        elif approach == 'Single Task':
            approach_fn = SingleTaskQLearningApproach
            file = 'single_task.png'
        elif approach == 'Multitask':
            approach_fn = MultiTaskQLearningApproach
            file = 'multitask.png'
        elif approach == 'Single Task with Hacking':
            approach_fn = SingleTaskAugmentedQLearningApproach
            file = 'multitask_augmented.png'
        elif approach == 'Multitask with Hacking':
            approach_fn = MultiTaskAugmentedQLearningApproach
            file = 'single_task_augmented.png'

        num_tasks = 30
        results = [0]*num_tasks
        for i in range(25):
            current = (test_single_approach(TaskGenerator, approach_fn))
            for j in range(num_tasks):
                results[j] += sum(current[j])/25.0


        # Data for plotting
        x = range(num_tasks)
        y = results

        fig, ax = plt.subplots()
        ax.plot(x,y)

        ax.set(xlabel='Number of Completed Tasks', ylabel='Return',
               title=str(approach))
        ax.grid()

        # fig.savefig(file)
        plt.show()






# plot returns
    # include random actions as baseline
        # shouldn't have a trend - noisy
        # single task should be better than this
    # plot episode vs total return for the episode
        # want to see that later episodes have higher returns
        # test that multitask is better than single when not randomizing
        # if multitask isnt improving, might be due to length/num of episodes 
        # or distribution of rewards (make range smaller)
    # save results using pickle
        # script that loads results and makes plots

# hacking reward function
    # ask reward fn rewards at empty space vs pellet, include in augmented state ([row, col, reward at pellet, reward at empty space])
        # dont randomize map
    
# try new approach with right hard coded policy
# unit tests