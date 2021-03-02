import numpy as np
# from meta_policy_search.envs.base import MetaEnv
    # MetaEnv not used, ignored it (prob just had it as parent class to ensure methods were implemented)
from meta_policy_search.utils import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv


class HalfCheetahRandDirecEnv(MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        # gym.utils.EzPickle.__init__(self, goal_direction) #??? how to set goal_direction? why isn't this redone in reset fn?
        gym.utils.EzPickle.__init__(self, 1.0)
        self.possible_targets = (-1.0, 1.0)

        self.state = [0, 0] # need to change
        self.done = False
        self.action_space = spaces.Discrete(2) # need to change
        self.observation_space = spaces.Box(-1000, 1000, shape=(2,), dtype=np.float32) # need to change
        self.max_steps = 25
        self.step_count = 0

    def randomize_rewards(self, rng):
        self.target = rng.choice(self.possible_targets)

        def rewards(state, action, next_state):
            pass
        # self.reward_function = self.target
        self.reward_function = rewards
        self.reward_function._target = self.target

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
        reward_run = self.goal_direction * (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset(self, rng):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    # def viewer_setup(self):
    #     self.viewer.cam.distance = self.model.stat.extent * 0.5

    # def log_diagnostics(self, paths, prefix=''):
    #     fwrd_vel = [path["env_infos"]['reward_run'] for path in paths]
    #     final_fwrd_vel = [path["env_infos"]['reward_run'][-1] for path in paths]
    #     ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

    #     logger.logkv(prefix + 'AvgForwardVel', np.mean(fwrd_vel))
    #     logger.logkv(prefix + 'AvgFinalForwardVel', np.mean(final_fwrd_vel))
    #     logger.logkv(prefix + 'AvgCtrlCost', np.std(ctrl_cost))

    # def __str__(self):
    #     return 'HalfCheetahRandDirecEnv'