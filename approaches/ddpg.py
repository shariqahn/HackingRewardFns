from .approach import Approach

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ddpg.core as core
from spinup.algos.pytorch.ddpg.ddpg import ReplayBuffer
from spinup.algos.pytorch.ddpg.core import MLPActor, MLPQFunction

from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs

from gym import spaces #fix



class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()

class SingleTaskDDPG(Approach):
    def __init__(self, action_space, observation_space, rng, eps=0.9, discount_factor=0.99, alpha=1e-3):
        self.rng = rng
        logger_kwargs = setup_logger_kwargs('SingleTaskDDPG', self.rng)
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(locals())

        self.actor_critic=MLPActorCritic
        # ac_kwargs=dict() ****?????*****
        # seed=0
        self.replay_size=int(1e6)
        self.polyak=0.995
        self.gamma=discount_factor
        self.pi_lr=alpha
        self.q_lr=alpha
        self.batch_size=100
        self.start_steps=10000
        self.update_after=1000
        self.update_every=50
        self.act_noise=0.1

        self.step_count = 0
        self.action_space = action_space
        self.observation_space = observation_space
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=(17,), dtype=np.float32) #fix

        # torch.manual_seed(seed)
        # np.random.seed(seed)

        # self.obs_dim = self.observation_space.shape
        self.act_dim = self.action_space.shape[0]
        # act_dim = self.action_space.n

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = self.action_space.high[0]

        self.net = False

    def init_net(self, state):
        self.obs_dim = state.shape
        # Create actor-critic module and target networks
        self.ac = self.actor_critic(self.obs_dim[0], self.action_space) #took out ac_kwargs
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=self.q_lr)
        self.logger.setup_pytorch_saver(self.ac)

        self.net = True

    def observe(self, state, action, next_state, reward, done):
        # use this transition to augment state properly - only the ifrst time
        state = self.process_state(state)
        next_state = self.process_state(next_state)

        self.replay_buffer.store(state, action, reward, next_state, done)
        if self.step_count >= self.update_after and self.step_count % self.update_every == 0:
            for _ in range(self.update_every):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                self.update(data=batch)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        self.logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, state, exploit=False):
        # for first call, will add random junk to processed state bc no transition yet
        processed_state = self.process_state(state)
        if not self.net:
            self.init_net(processed_state)

        # state is actually observation
        self.step_count += 1
        if self.step_count <= self.start_steps:
            return self.action_space.sample()

        a = self.ac.act(torch.as_tensor(processed_state, dtype=torch.float32))
        if not exploit:
            a += self.act_noise * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def reset(self, reward_function):
        self.reward_function = reward_function
        self.net = False
        # self.step_count = 0

    def process_state(self, state):
        return state

    def log(self, returns, task):
        self.logger.store(EpRet=sum(returns), EpLen=len(returns))
        self.logger.save_state({'env': task}, None)
        self.logger.log_tabular('EpRet', with_min_and_max=True)
        self.logger.log_tabular('EpLen', average_only=True)
        self.logger.log_tabular('TotalEnvInteracts', self.step_count)
        self.logger.log_tabular('QVals', with_min_and_max=True)
        self.logger.log_tabular('LossPi', average_only=True)
        self.logger.log_tabular('LossQ', average_only=True)
        self.logger.dump_tabular()


class MultiTaskDDPG(SingleTaskDDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger_kwargs = setup_logger_kwargs('MultiTaskDDPG')
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(globals())

        self.start_steps = 10000

    def reset(self, reward_function):
        self.reward_function = reward_function


class MultiTaskDDPGAugmentedOracle(MultiTaskDDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger_kwargs = setup_logger_kwargs('MultiTaskDDPGAugmentedOracle')
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(globals())

    def process_state(self, state):
        query = self.reward_function(None, None, None, True)
        return np.append(state, query)


class MultiTaskDDPGQuery(MultiTaskDDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger_kwargs = setup_logger_kwargs('MultiTaskDDPGQuery')
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(globals())

    def process_state(self, state):
        query_state = np.zeros(17)
        query_state[0] = -.05
        next_state = np.zeros(17)
        next_state[0] = .007
        action = self.rng.rand(6)
        query = self.reward_function(query_state, action, next_state) 
        return np.append(state, query)


class MultiTaskDDPGAutoQuery(MultiTaskDDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger_kwargs = setup_logger_kwargs('MultiTaskDDPGQuery')
        self.logger = EpochLogger(**logger_kwargs)
        self.logger.save_config(globals())

    def process_state(self, state):
        query_state = np.zeros(17)
        query_state[0] = -.05
        next_state = np.zeros(17)
        next_state[0] = .007
        action = self.rng.rand(6)
        query = self.reward_function(query_state, action, next_state) 
        return np.append(state, query)

