import gym
import gym_slider
import numpy as np


def calculate_max_returns():
    env = gym.make('slider-v0')
    max_returns = {}

    for i in range(-50,50):
        rng = np.random.RandomState(np.random.randint(0,100))
        env.reset(rng)
        env.target_velocity = i
        done = False
        returns = 0
        count = 0
        while not done:
            count += 1
            if env.target_velocity > 0:
                action = 1
            else:
                action = 0
            x = env.step(action)
            # print(x)
            done = x[2]
            returns += x[1]
            # print(x[1])
        # print(count)
        max_returns[i] = returns

    return max_returns
        # print(env.target_velocity, returns)

    # print(max_returns)

max_returns = calculate_max_returns()

