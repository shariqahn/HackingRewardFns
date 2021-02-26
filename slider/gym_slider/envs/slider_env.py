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
        self.reward_scale = 1.0
        self.possible_targets = range(-10, 11)

        self.viewer = None
        self.state = [0, 0]
        self.done = False
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-1000, 1000, shape=(2,), dtype=np.float32)
        self.max_steps = 25
        self.step_count = 0

    def randomize_rewards(self, rng):
        self.target_velocity = rng.choice(self.possible_targets)

        def rewards(state):
            if state[1] == self.target_velocity:
                return 10
            return -1 * self.reward_scale * (self.target_velocity - state[1])**2
        # self.reward_function = self.target_velocity
        self.reward_function = rewards
        self.reward_function._target_velocity = self.target_velocity

    def step(self, action):
        x, x_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        xacc = force / self.mass

        x_dot = x_dot + self.tau * xacc

        x = x + self.tau * x_dot

        self.state = [x, x_dot]

        self.done = bool(
            x_dot > (self.target_velocity - 3)
            and x_dot < (self.target_velocity + 3)
        )

        self.step_count += 1

        reward = self.reward_function(self.state)

        if self.step_count >= self.max_steps:
            self.done = True

        return np.array(self.state), reward, self.done, {}

    def reset(self, rng):
        self.randomize_rewards(rng)
        self.state = [0,0]
        self.done = False
        self.step_count = 0
        return np.array(self.state)

    def render(self, mode='human'):
        pass

    # def close(self):
    #     if self.viewer:
    #         self.viewer.close()
    #         self.viewer = None
