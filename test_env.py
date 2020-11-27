import gym
import gym_slider
import numpy as np
env = gym.make('slider-v0')

rng = np.random.RandomState(0)
env.reset(rng)

done = False
while not done:
    x = env.step(1)
    print(x)
    done = x[2]
