import gym
import gym_foo
import matplotlib.pyplot as plt
# import ipdb

from approaches.random_policy import RandomPolicyApproach
from approaches.q_learning import SingleTaskQLearningApproach, MultiTaskQLearningApproach, SingleTaskAugmentedQLearningApproach, MultiTaskAugmentedQLearningApproach

def test_single_approach(task_generator, approach, total_num_tasks=30):
    results = []
    task = task_generator('foo-v0')
    approach = approach(task.get_actions(), task.get_rewards)

    for task_num in range(total_num_tasks):
        result = run_approach_on_task(approach, task)
        results.append(result)

    return results

def run_approach_on_task(approach, task, max_num_steps=100):
    # Task is a Gym env
    state = task.reset()
    # Initialize the approach
    approach.reset()
    # Result is a list of all rewards seen
    result = []
    for t in range(max_num_steps):
        action = approach.get_action(state)
        next_state, reward, done, _ = task.step(action)
        result.append(reward)
        # Tell the approach about the transition (for learning)
        approach.observe(state, action, next_state, reward, done)
        state = next_state
        if done:
            # break
            state = task.reset()
            # reset environment without rerandomizing
            # allows agent to learn for longer
    return result

class TaskGenerator:
    def __init__(self, env_name):
        self.env = gym.make(env_name)

    def reset(self):
        # self.env.randomize_map()
        self.env.randomize_rewards()
        state = self.env.reset()
        return state

    def step(self, action):
        return self.env.step(action)

    def get_actions(self):
        return self.env.action_space

    def get_rewards(self):
        return self.env.reward_function


if __name__ == "__main__":
    for approach in ('Random Policy', 'Single Task', 'Multitask', 'Single Task with Hacking', 'Multitask with Hacking'):
        if approach == 'Random Policy':
            approach_fn = RandomPolicyApproach
            file = 'random.png'
        elif approach == 'Single Task':
            approach_fn = SingleTaskQLearningApproach
            file = 'single_task.png'
        elif approach == 'Multitask':
            approach_fn = MultiTaskQLearningApproach
            file = 'multitask.png'
        elif approach == 'Single Task with Hacking':
            approach_fn = SingleTaskAugmentedQLearningApproach
            file = 'multitask_augmented.png'
        elif approach == 'Multitask with Hacking':
            approach_fn = MultiTaskAugmentedQLearningApproach
            file = 'single_task_augmented.png'

        num_tasks = 30
        results = [0]*num_tasks
        for i in range(25):
            current = (test_single_approach(TaskGenerator, approach_fn))
            for j in range(num_tasks):
                results[j] += sum(current[j])/25.0


        # Data for plotting
        x = range(num_tasks)
        y = results

        fig, ax = plt.subplots()
        ax.plot(x,y)

        ax.set(xlabel='Number of Completed Tasks', ylabel='Return',
               title=str(approach))
        ax.grid()

        # fig.savefig(file)
        plt.show()

# plot returns
    # include random actions as baseline
        # shouldn't have a trend - noisy
        # single task should be better than this
    # plot episode vs total return for the episode
        # want to see that later episodes have higher returns
        # test that multitask is better than single when not randomizing
        # if multitask isnt improving, might be due to length/num of episodes 
        # or distribution of rewards (make range smaller)
    # save results using pickle
        # script that loads results and makes plots

# hacking reward function
    # ask reward fn rewards at empty space vs pellet, include in augmented state ([row, col, reward at pellet, reward at empty space])
        # dont randomize map
    
# try new approach with right hard coded policy
# unit tests

