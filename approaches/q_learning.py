from .approach import Approach
from collections import defaultdict
import numpy as np
from random import randint

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