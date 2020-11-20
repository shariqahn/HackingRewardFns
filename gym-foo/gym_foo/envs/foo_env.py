import gym
from gym import error, spaces, utils
import numpy as np
from random import randint, choice
# import ipdb

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.map = [['','','','.'],['.','.','.','.'],['','','.','.'],['.','','','G']]
        self.state = [0,0]
        self.reward = 0
        self.done = False
        self.action_space = ['right', 'left', 'up', 'down'] #use openai action spaces instead!!!
        # self.randomize_rewards()

    def randomize_rewards(self, rng): 
        all_possible_rewards = [-50, -1]
        rng.shuffle(all_possible_rewards)
        pellet_reward, space_reward = all_possible_rewards[0], all_possible_rewards[1]

        self.pellet_reward = pellet_reward
        self.space_reward = space_reward

        rewards = {}
        for i in range(len(self.map)):
            for j in range(len(self.map[0])):
                if self.map[i][j] == '.':
                    rewards[str([i,j])] = pellet_reward
                elif self.map[i][j] == '':
                    rewards[str([i,j])] = space_reward
                else:
                    rewards[str([i,j])] = 10

        self.reward_function = rewards
        return rewards

    def randomize_map(self):
        new_map = [['','','',''],['','','',''],['','','',''],['','','','']]
        for i in range(8):
            row = randint(0,3)
            column = randint(0,3)
            while new_map[row][column] == '.':
              row = randint(0,3)
              column = randint(0,3)
            new_map[row][column] = '.'

        goal_row = randint(0,3)
        goal_column = randint(0,3)
        while new_map[goal_row][goal_column] == '.':
            goal_row = randint(0,3)
            goal_column = randint(0,3)
        new_map[goal_row][goal_column] = 'G'  

        self.map = new_map
        return new_map

    def step(self, action):
        if action == 'right':
            self.state[1] = (self.state[1] + 1) % 4
        elif action == 'left':
            self.state[1] = (self.state[1] + 3) % 4
        elif action == 'up':
            self.state[0] = (self.state[0] + 3) % 4
        else:
            self.state[0] = (self.state[0] + 1) % 4

        reward = self.reward_function[str([self.state[0], self.state[1]])]
        self.reward += reward
        if self.map[self.state[0]][self.state[1]] == "G":
            self.done = True

        return str(self.state), reward, self.done, self.map

    def reset(self, rng):
        self.randomize_rewards(rng)
        self.state = [0,0] 
        self.reward = 0
        self.done = False
        return str(self.state)

    def render(self, mode='human', close=False):
        print(self.state, self.reward)