import gym
import gym_slider
import numpy as np
env = gym.make('slider-v0')

rng = np.random.RandomState(0)
env.reset()

done = False
returns = 0
while not done:
    x = env.step(0)
    print(x)
    done = x[2]
    returns += x[1]

print(returns)
