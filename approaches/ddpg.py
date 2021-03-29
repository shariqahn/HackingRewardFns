from .approach import Approach

from copy import deepcopy
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.ddpg.core as core
from spinup.algos.pytorch.ddpg.ddpg import ReplayBuffer

from gym import spaces #fix

class SingleTaskDDPG(Approach):
    def __init__(self, action_space, observation_space, rng, eps=0.9, discount_factor=0.99, alpha=1e-3):
        self.actor_critic=core.MLPActorCritic
        # ac_kwargs=dict() ****?????*****
        # seed=0
        self.replay_size=int(1e6)
        polyak=0.995
        gamma=discount_factor
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

        self.obs_dim = self.observation_space.shape
        self.act_dim = self.action_space.shape[0]
        # act_dim = self.action_space.n

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.action_space.high[0]

        # Create actor-critic module and target networks
        ac = self.actor_critic(self.observation_space, self.action_space) #took out ac_kwargs
        ac_targ = deepcopy(ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(ac.q.parameters(), lr=self.q_lr)

    def observe(self, state, action, next_state, reward, done):
        self.replay_buffer.store(state, action, reward, next_state, done)
        if self.step_count >= self.update_after and self.step_count % self.update_every == 0:
            for _ in range(self.update_every):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                self.update(data=batch)

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o,a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup)**2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = ac.q(o, ac.pi(o))
        return -q_pi.mean()

    def update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort 
        # computing gradients for it during the policy learning step.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in ac.q.parameters():
            p.requires_grad = True

        # Record things
        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(self, state, exploit=False):
        # state is actually observation
        self.step_count += 1
        if self.step_count <= self.start_steps:
            return self.action_space.sample()

        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += self.act_noise * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def reset(self, reward_function):
        self.reward_function = reward_function

        ac = self.actor_critic(self.observation_space, self.action_space)
        ac_targ = deepcopy(ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in ac_targ.parameters():
            p.requires_grad = False

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=self.replay_size)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(ac.pi.parameters(), lr=self.pi_lr)
        self.q_optimizer = Adam(ac.q.parameters(), lr=self.q_lr)
