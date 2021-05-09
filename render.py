import torch
import numpy as np

from cheetah import HalfCheetah
from approaches.ddpg import SingleTaskDDPG, MultiTaskDDPG, MultiTaskDDPGAugmentedOracle, MultiTaskDDPGQuery


if __name__ == '__main__':
    task = HalfCheetah(render=True)
    rng = np.random.RandomState(2)
    task.reset(rng)

    approach = MultiTaskDDPGAugmentedOracle(task.action_space, task.observation_space, rng)
    approach.reset(task.reward_function)
    approach.start_steps = 0

    print(approach.load('/Users/shariqah/spinningup/data/MultiTaskDDPGAugmentedOracle/pyt_save/model.pt', task))
