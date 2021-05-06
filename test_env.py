import gym
import gym_slider
import numpy as np
import pybulletgym
from cheetah import HalfCheetah


def calculate_max_returns():
    env = gym.make('slider-v0')
    max_returns = {}

    for i in range(-50,50):
        rng = np.random.RandomState(np.random.randint(0,100))
        env.reset(rng)
        env.target_velocity = i
        done = False
        returns = 0
        num_steps = 25
        step_count = 0
        while not done:
            step_count += 1
            if env.target_velocity > 0:
                action = 1
            else:
                action = 0
            x = env.step(action)
            # print(x)
            done = x[2]
            returns += x[1]
            if step_count == num_steps:
                done = True
                step_count = 0
            # print(x[1])
        max_returns[i] = returns

    return max_returns
        # print(env.target_velocity, returns)

# max_returns = calculate_max_returns()
# print(max_returns)
env = HalfCheetah()
print(env.observation_space)

# env = HalfCheetah()
# rng = np.random.RandomState(0)
# env.reset(rng)
# r = 0
# while(True):
#     action = np.random.normal(size = env.action_space.shape[0])
#     state, reward, done, _ = env.step(action)
#     r += reward
#     if done:
#         print("Random: Reward at Termination: {}".format(r))
#         break

images = []
im = task.render(mode='rgb_array')
images.append(im)
imageio.mimsave(images, [path to file])