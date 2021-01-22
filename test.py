import gym
import gym_foo
import matplotlib.pyplot as plt
import numpy as np
# import ipdb

from approaches.random_policy import RandomPolicyApproach
from approaches.q_learning import SingleTaskQLearningApproach, MultiTaskQLearningApproach, SingleTaskAugmentedQLearningApproach, MultiTaskAugmentedQLearningApproach
from approaches.dqn import SingleTaskDQN, MultiTaskDQN

def test_single_approach(approach, rng, total_num_tasks=40):
    results = []
    task = gym.make('foo-v0')
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
    # count = 0
    # A list of all the states ever visited
    # all_states = [state]
    results = []
    while len(results) < num_tasks:
        action = approach.get_action(state)
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
            # reset environment without rerandomizing
            # allows agent to learn for longer
            # count += 1
            results.append(result)
            result = []
    return results
    # , count
    # , all_states


if __name__ == "__main__":
    maximums = []
    minimums = []
    for approach in ('Random Policy', 'Single Task', 'Multitask', 'Single Task with Hacking', 'Multitask with Hacking'):
        # 'SingleTaskDQN', 'MultiTaskDQN'):
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
            file = 'single_dqn.png'
        elif approach == 'MultiTaskDQN':
            approach_fn = MultiTaskDQN
            file = 'multi_dqn.png'

        num_tasks = 40
        results = [0]*num_tasks
        goals = [0]*num_tasks
        # state_visits = np.zeros((4, 4))
        for i in range(25):
            rng = np.random.RandomState(i)
            # rng random num generator
            current = (test_single_approach(approach_fn, rng))
            for j in range(num_tasks):
                results[j] += sum(current[j])/25.0
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
        # plt.show()

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
