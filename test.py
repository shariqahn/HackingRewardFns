import gym
import gym_foo
import gym_slider
import matplotlib.pyplot as plt
import numpy as np
import torch

from approaches.random_policy import RandomPolicyApproach
from approaches.q_learning import SingleTaskQLearningApproach, MultiTaskQLearningApproach, SingleTaskAugmentedQLearningApproach, MultiTaskAugmentedQLearningApproach
from approaches.dqn import SingleTaskDQN, MultiTaskDQN, SingleTaskAugmentedDQN, MultiTaskAugmentedDQN

from test_env import max_returns

num_tasks = 100
def test_single_approach(approach, rng, total_num_tasks=num_tasks):
    results = []
    # task = gym.make('foo-v0')
    task = gym.make('slider-v0')
    approach = approach(task.action_space, rng)

    results = run_approach_on_task(approach, task, rng, num_tasks=total_num_tasks)
    return results

def run_approach_on_task(approach, task, rng, num_tasks):
    # Task is a Gym env
    state = task.reset(rng)
    # Initialize the approach
    approach.reset(task.reward_function)
    # Result is a list of all rewards seen
    result = []
    actions = []
    # count = 0
    # A list of all the states ever visited
    # all_states = [state]
    results = []
    differences = []
    targets = [task.target_velocity]
    while len(results) < num_tasks:
        action = approach.get_action(state)
        actions.append(action)
        next_state, reward, done, _ = task.step(action)
        result.append(reward)
        # all_states.append(next_state)
        # Tell the approach about the transition (for learning)
        approach.observe(state, action, next_state, reward, done)
        state = next_state
        if done:
            # break
            state = task.reset(rng)
            approach.reset(task.reward_function)
            targets.append(task.target_velocity)
            # reset environment without rerandomizing
            # allows agent to learn for longer
            # count += 1
            results.append(result)
            diff = -1
            if (actions[0] == 0 and task.target_velocity <= 0) or (actions[0] == 1 and task.target_velocity >= 0):
                diff += 1
            consistent = True
            for a in actions:
                if a != actions[0]:
                    consistent = False
            if consistent:
                diff += 1 
            differences.append(diff)
            result = []
            actions = []
    return results, differences, targets[:-1]
    # , count
    # , all_states


if __name__ == "__main__":
    maximums = []
    minimums = []
    for approach in ('SingleTaskDQN', 'MultiTaskDQN', 'SingleTaskAugmentedDQN', 'MultiTaskAugmentedDQN'):
        print(approach)
    # ('Random Policy', 'Single Task', 'Multitask', 'Single Task with Hacking', 'Multitask with Hacking'):
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
            file = 'single_task_augmented.png'
        elif approach == 'Multitask with Hacking':
            approach_fn = MultiTaskAugmentedQLearningApproach
            file = 'multitask_augmented.png'
        elif approach == 'SingleTaskDQN':
            approach_fn = SingleTaskDQN
            file = 'updated_reward_single_dqn.png'
        elif approach == 'MultiTaskDQN':
            approach_fn = MultiTaskDQN
            file = 'updated_reward_multi_dqn.png'
        elif approach == 'SingleTaskAugmentedDQN':
            approach_fn = SingleTaskAugmentedDQN
            file = 'updated_reward_single_augmented_dqn.png'
        elif approach == 'MultiTaskAugmentedDQN':
            approach_fn = MultiTaskAugmentedDQN
            file = 'updated_reward_multi_augmented_dqn.png'

        results = [0]*num_tasks
        goals = [0]*num_tasks 
        
        # state_visits = np.zeros((4, 4))
        for i in range(25):
            if i == 0:
                difference = [0]*num_tasks #difference bw returns and target velocities

            rng = np.random.RandomState(i)
            torch.manual_seed(i)
            
            rewards, differences, targets = (test_single_approach(approach_fn, rng))

            for j in range(num_tasks):
                episode_return = sum(rewards[j])
                results[j] += episode_return/25.0

                # goals[j] += current[j][1]/25.0
                # for s in current[j][2]:
                #     r, c = eval(s)
                #     state_visits[r, c] += 1./25.0


        maximums.append(max(results))
        minimums.append(min(results))


        # Data for plotting
        x = range(num_tasks)
        y = results

        fig, ax = plt.subplots()
        ax.plot(x,y)

        ax.set(xlabel='Number of Episodes', ylabel='Return',
               title=approach)
        # ax.set_ylim(ymax=40,ymin=-300)
        ax.grid()

        fig.savefig(file)

        # plot with max return reference line
        plt.clf()
        z = []
        final_results = [] # returns from 25th test
        for i in range(len(rewards)):
            target = targets[i]
            for j in range(len(rewards[i])):
                final_results.append(rewards[i][j])
                z.append(max_returns[target])

        x = range(1,len(final_results)+1)
        plt.plot(x,final_results, label = 'results')
        plt.plot(x,z, label = 'max returns')

        plt.xlabel('Number of Steps')
        plt.ylabel('Reward')
        plt.title(approach)
        # ax.set_ylim(ymax=40,ymin=-300)
        # ax.grid()
        plt.legend()
        plt.savefig(file[:-4] + '_reference.png')

        # plot of scores
        x = range(num_tasks)
        z = differences

        fig, ax = plt.subplots()
        ax.plot(x,z)

        ax.set(xlabel='Number of Episodes', ylabel='Policy Scores',
               title=approach)
        ax.grid()

        fig.savefig(file[:-4]+'_scores.png')

        # y = goals

        # fig, ax = plt.subplots()
        # ax.plot(x,y)

        # ax.set(xlabel='Number of Episodes', ylabel='Number of Times Goal Reached',
        #        title=approach)
        # ax.grid()

        # fig.savefig(file[:-4]+'_goals.png')

        # Plot state visits
        # fig, ax = plt.subplots()
        # ax.set(title=approach)
        # ax.axis('off')
        # plt.imshow(state_visits, vmin=0, cmap='jet')
        # plt.colorbar()
        # fig.savefig(file[:-4]+'_state_visits.png')


    print("max: ", maximums)
    print("min: ", minimums)
