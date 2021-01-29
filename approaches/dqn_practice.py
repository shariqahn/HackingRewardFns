import gym
import gym_slider
import numpy as np

from stable_baselines import DQN

env = gym.make('slider-v0')
# env = gym.make('LunarLander-v2')

model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

def evaluate(model, num_steps=1000):
	"""
	Evaluate a RL agent
	:param model: (BaseRLModel object) the RL Agent
	:param num_steps: (int) number of timesteps to evaluate it
	:return: (float) Mean reward for the last 100 episodes
	"""
	episode_rewards = [0.0]
	obs = env.reset()
	for i in range(num_steps):
	  # _states are only useful when using LSTM policies
	  action, _states = model.predict(obs)
	  # print(action)
	  obs, reward, done, info = env.step(action)
	  
	  # Stats
	  episode_rewards[-1] += reward
	  if done:
	      obs = env.reset()
	      episode_rewards.append(0.0)
	# Compute mean reward for the last 100 episodes
	mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
	print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

	return mean_100ep_reward


mean_reward_before_train = evaluate(model)

# Train the agent
model.learn(total_timesteps=int(2e4), log_interval=10)

# # Save the agent
# model.save("dqn_slider")
# del model  # delete trained model to demonstrate loading

# model = DQN.load("dqn_slider")

# Evaluate the trained agent
mean_reward = evaluate(model)

# model.learn(total_timesteps=int(1e4))
# mean_reward = evaluate(model, num_steps=500)