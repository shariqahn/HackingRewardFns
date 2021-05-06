import gym
from gym import spaces, logger
from gym.utils import seeding

import numpy as np
from environment import Environment
from pybulletgym.envs.mujoco.envs.locomotion.half_cheetah_env import HalfCheetahMuJoCoEnv

import imageio

class HalfCheetah(Environment, HalfCheetahMuJoCoEnv):
    def __init__(self):
        HalfCheetahMuJoCoEnv.__init__(self)

        self.possible_targets = (-1.0, 1.0)
        self.target = 1.0
        # self.state = [0, 0] # need to change
        # self.done = False
        # self.action_space = spaces.Discrete(2) # need to change
        # self.observation_space = spaces.Box(-1000, 1000, shape=(2,), dtype=np.float32) # need to change

        # better to handle this in testing - do this w other env
        # self.max_steps = 25
        # self.step_count = 0

        self.dt = 1/240

    def randomize_rewards(self, rng):
        self.target = rng.choice(self.possible_targets)
        # self.target = 1.0
        self.reward_function = self.get_reward

    def get_reward(self, state, action, next_state, target=False):
        # state is vector where first elem is xpos
        if target:
            return self.target

        xposbefore = state[0]
        xposafter = next_state[0]
        reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
        reward_run = self.target * (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        
        return reward

    def get_state(self):
        qpos = np.array([j.get_position() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)
        qvel = np.array([j.get_velocity() for j in self.ordered_joints], dtype=np.float32).flatten()  # shape (9,)

        return np.concatenate([
            qpos.flat,           # self.sim.data.qpos.flat[1:],
            qvel.flat                # self.sim.data.qvel.flat,
        ])

    def step(self, action):
        state = self.get_state()

        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(action)
            self.scene.global_step()

        next_state = self.get_state()
        self.reward = self.get_reward(state, action, next_state)
        obs = self.robot.calc_state()
        done = False

        self.HUD(obs, action, done)

        return obs, self.reward, done, {}

    def reset(self, rng):
        self.randomize_rewards(rng)

        return HalfCheetahMuJoCoEnv.reset(self)

    # def reset(self):
    #     return HalfCheetahMuJoCoEnv.reset(self)

if __name__ == '__main__':
    env = HalfCheetah()
    rng = np.random.RandomState(0)
    state = env.reset(rng)
    # action = env.action_space.sample()
    action = rng.rand(6)
    next_state, _, _, _ = env.step(action)
    print(state, action, next_state)

    # s = np.zeros(17)
    # s[0] = -.056
    # print(s.shape == state.shape)
    # a = 

    # print(env.observation_space.shape)
    # imgs = []
    # for i in range(100):
    #     obs,_,_,_ = env.step(env.action_space.sample())
    #     # aug = (np.append(obs, 150))
    #     # print(aug)
    #     # print('len', (aug.shape))
    #     image = env.render(mode='rgb_array')
    #     imgs.append(image)

    # imageio.mimsave('video.mp4', imgs)


    
