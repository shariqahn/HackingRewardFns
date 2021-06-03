from .approach import Approach
from collections import defaultdict
import numpy as np
# import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, num_actions)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class SingleTaskDQN(Approach):
    def __init__(self, action_space, rng, eps=0.9, discount_factor=0.9, alpha=0.01):
        self.action_space = action_space
        self.eps = eps # probability of random action
        self.discount_factor = discount_factor # gamma
        self.alpha = alpha # learning rate

        self.rng = rng 
        self.batch_size = 32
        self.env_shape = 0 if isinstance(action_space.sample(), int) else action_space.sample().shape
        self.num_actions = action_space.n
        # self.num_actions = action_space.shape[0]
        self.memory_capacity = 2000
        self.net = False

    def init_net(self, state):
        self.num_states = len(state)

        self.eval_net, self.target_net = Net(self.num_states, self.num_actions), Net(self.num_states, self.num_actions)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((self.memory_capacity, self.num_states * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.alpha)
        self.loss_func = nn.MSELoss()

        self.net = True

    def reset(self, reward_function):
        self.reward_function = reward_function
        self.net = False

    def get_action(self, state, exploit=False):
        processed_state = self.process_state(state)
        if not self.net:
            self.init_net(processed_state)
        state = torch.unsqueeze(torch.FloatTensor(processed_state), 0)
        # input only one sample
        if exploit or (self.rng.uniform() < self.eps):   # greedy
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if self.env_shape == 0 else action.reshape(self.env_shape)  # return the argmax index
        else:   # random
            action = self.rng.randint(0, self.num_actions)
            action = action if self.env_shape == 0 else action.reshape(self.env_shape)
        return action

    def observe(self, state, action, next_state, reward, done):
        state = self.process_state(state)
        next_state = self.process_state(next_state)
        transition = np.hstack((state, [action, reward], next_state))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
        if self.memory_counter > self.memory_capacity:
            self.learn()

    def learn(self):
        # target parameter update
        target_replace_iter = 100
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = self.rng.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.num_states])
        b_a = torch.LongTensor(b_memory[:, self.num_states:self.num_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.num_states+1:self.num_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.num_states:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + self.discount_factor * q_next.max(1)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def process_state(self, state):
        return state


class MultiTaskDQN(SingleTaskDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, reward_function):
        self.reward_function = reward_function


class SingleTaskAugmentedDQN(SingleTaskDQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_state(self, state):
        query = self.reward_function(None, None, None, True)
        # ipdb.set_trace()
        return np.append(state, query)
        

# class MultiTaskAugmentedDQN(SingleTaskAugmentedDQN):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def reset(self, reward_function):
#         self.reward_function = reward_function


class MultiTaskAugmentedOracle(MultiTaskDQN):
    def process_state(self, state):
        query = self.reward_function(None, None, None, True)
        # ipdb.set_trace()
        return np.append(state, query)


class MultiTaskDQNOneQuery(MultiTaskDQN):
    def process_state(self, state):
        query = self.reward_function(np.array([1,1]),1, np.array([2,2])) 
        # ipdb.set_trace()
        return np.append(state, query)


class MultiTaskDQNTwoQuery(MultiTaskDQN):
    def process_state(self, state):
        query = [self.reward_function(np.array([1,1]),1, np.array([2,2])), self.reward_function(np.array([-1,-1]),1, np.array([0,0]))]

        # Debug: check that we can recover the target velocity
        # if query[0] == query[1]:
        #     v = 0
        # else:
        #     v = np.mean(np.sqrt(-np.array(query)))
        #     if query[0] < query[1]:
        #         v *= -1
        # assert v == self.reward_function._target

        # ipdb.set_trace()
        return np.append(state, query)

