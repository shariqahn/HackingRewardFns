import torch
import numpy as np

from cheetah import HalfCheetah
from approaches.ddpg import SingleTaskDDPG, MultiTaskDDPG, MultiTaskDDPGAugmentedOracle, MultiTaskDDPGQuery


if __name__ == '__main__':
    task = HalfCheetah()
    rng = np.random.RandomState(2)
    task.reset(rng)
    approach = MultiTaskDDPGAugmentedOracle(task.action_space, task.observation_space, rng)
    approach.reset(task.reward_function)

    # model = torch.load('/Users/shariqah/spinningup/data/MultiTaskDDPG/pyt_save/model.pt')
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # model.eval()
    print(approach.load('/Users/shariqah/spinningup/data/MultiTaskDDPGAugmentedOracle/pyt_save/model.pt', task))
