from .approach import Approach
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