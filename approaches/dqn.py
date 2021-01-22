from .approach import Approach
from collections import defaultdict
import numpy as np

class SingleTaskDQN(Approach):

    def __init__(self, action_space, rng, eps=0.1, discount_factor=0.9, alpha=0.5):
        self.action_space = action_space
        # self.eps = eps # probability of random action
        # self.discount_factor = discount_factor # gamma
        self.alpha = alpha # learning rate
        self.rng = rng

    def reset(self, reward_function):
        self.reward_function = reward_function
        self.model = DQN('MlpPolicy', env, learning_rate=self.alpha, prioritized_replay=True, verbose=0)

    def get_action(self, state):
        return self.predict(state)[0]

    def observe(self, state, action, next_state, reward, done):
        self.model.learn(total_timesteps=10, log_interval=10)


class MultiTaskDQN(SingleTaskDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DQN('MlpPolicy', env, learning_rate=self.alpha, prioritized_replay=True, verbose=0)

    def reset(self, reward_function):
        self.reward_function = reward_function