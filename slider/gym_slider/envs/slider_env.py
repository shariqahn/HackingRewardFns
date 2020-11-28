import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class SliderEnv(gym.Env):
    # metadata = {
    #     'render.modes': ['human', 'rgb_array'],
    #     'video.frames_per_second': 50
    # }

    def __init__(self):
        self.mass = .1
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        self.target_velocity = 200

        self.viewer = None
        self.state = [0, 0]
        self.done = False
        self.action_space = [0, 1]

    def randomize_rewards(self, rng):
    	self.reward_function = 1


    def step(self, action):
        x, x_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        xacc = force / self.mass

        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot

        self.state = [x, x_dot]

        self.done = bool(
            x > (self.target_velocity - 3)
            and x < (self.target_velocity + 3)
        )

        if self.done:
            reward = self.reward_function
        else:
        	reward = 0

        return str(self.state), reward, self.done, ""

    def reset(self, rng):
        self.randomize_rewards(rng)
        self.state = [0,0]
        self.done = False
        return str(self.state)

    def render(self, mode='human'):
        pass

    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None