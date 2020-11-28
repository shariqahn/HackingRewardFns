from .approach import Approach
import numpy as np

class RandomPolicyApproach(Approach):

    def __init__(self, action_space, rng):
        self.action_space = action_space
        self.rng = rng

    def reset(self, reward_function):
        pass

    def observe(self, state, action, next_state, reward, done):
        pass

    def get_action(self, state):
        # return self.action_space.sample()
        # index = randint(0,3) # hacky! fix action space
        return self.action_space[self.rng.randint(0,4)]
        # return self.action_space[index]