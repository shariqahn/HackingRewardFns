import gym
import gym_foo
import gym_slider
import numpy as np
import torch
import pickle

from approaches.random_policy import RandomPolicyApproach
from approaches.q_learning import SingleTaskQLearningApproach, MultiTaskQLearningApproach, SingleTaskAugmentedQLearningApproach, MultiTaskAugmentedQLearningApproach
from approaches.dqn import MultiTaskAugmentedOracle, SingleTaskDQN, MultiTaskDQN, SingleTaskAugmentedDQN, MultiTaskDQNOneQuery, MultiTaskDQNTwoQuery


num_tasks = 600
eval_interval = 10
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
    results = [] # includes e-greedy results
    differences = []
    targets = [task.target]
    eval_results = [] # no e-greedy
    step_count = 0
    max_steps = 25
    while len(results) < num_tasks:
        if len(results) % eval_interval == 0:
            action = approach.get_action(state, True)
        else:
            action = approach.get_action(state)
        actions.append(action)
        next_state, reward, done, _ = task.step(action)
        step_count += 1
        result.append(reward)
        # all_states.append(next_state)
        # Tell the approach about the transition (for learning)
        approach.observe(state, action, next_state, reward, done)
        state = next_state

        if not done:
            done = step_count == max_steps

        if done:
            # break
            state = task.reset(rng)
            approach.reset(task.reward_function)
            step_count = 0
            targets.append(task.target)
            # reset environment without rerandomizing
            # allows agent to learn for longer
            # count += 1
            if len(results) % eval_interval == 0:
                eval_results.append(result)
                diff = 0
                if (actions[0] == 0 and task.target <= 0) or (actions[0] == 1 and task.target >= 0):
                    diff += 1
                consistent = 0
                for a in actions:
                    if a == actions[0]:
                        consistent += 1
                diff += consistent/len(actions)
                differences.append(diff)

            if len(results) % 10*eval_interval == 0:
                print(f"Finished trial {len(results)}/{num_tasks} with returns {sum(result):6.0f}", end='\r')
            results.append(result)

            result = []
            actions = []
    print()
    return eval_results, differences, targets[:-1]
    # , count
    # , all_states


if __name__ == "__main__":
    for approach in (
        'SingleTaskDQN',
        'SingleTaskAugmentedDQN',
        # 'MultiTaskAugmentedOracle',
        # 'MultiTaskDQNOneQuery',
        # 'MultiTaskDQNTwoQuery',
        # 'MultiTaskDQN'
        ):
        print(approach)
    # ('Random Policy', 'Single Task', 'Multitask', 'Single Task with Hacking', 'Multitask with Hacking'):
        file = 'results/'
        # if approach == 'Random Policy':
        #     approach_fn = RandomPolicyApproach
        #     file += 'random.pkl'
        # elif approach == 'Single Task':
        #     approach_fn = SingleTaskQLearningApproach
        #     file += 'single_task.pkl'
        # elif approach == 'Multitask':
        #     approach_fn = MultiTaskQLearningApproach
        #     file += 'multitask.pkl'
        # elif approach == 'Single Task with Hacking':
        #     approach_fn = SingleTaskAugmentedQLearningApproach
        #     file += 'single_task_augmented.pkl'
        # elif approach == 'Multitask with Hacking':
        #     approach_fn = MultiTaskAugmentedQLearningApproach
        #     file += 'multitask_augmented.pkl'
        if approach == 'SingleTaskDQN':
            approach_fn = SingleTaskDQN
            file += '_eval_single_dqn.pkl'
        elif approach == 'MultiTaskDQN':
            approach_fn = MultiTaskDQN
            file += '_eval_multi_dqn.pkl'
        elif approach == 'SingleTaskAugmentedDQN':
            approach_fn = SingleTaskAugmentedDQN
            file += '_eval_single_augmented_dqn.pkl'
        elif approach == 'MultiTaskAugmentedOracle':
            approach_fn = MultiTaskAugmentedOracle
            file += '_eval_MultiTaskAugmentedOracle.pkl'
        elif approach == 'MultiTaskDQNOneQuery':
            approach_fn = MultiTaskDQNOneQuery
            file += '_eval_multi_augmented_dqn_1_query.pkl'
        elif approach == 'MultiTaskDQNTwoQuery':
            approach_fn = MultiTaskDQNTwoQuery
            file += '_eval_multi_augmented_dqn_2_query.pkl'

        final_num_tasks = num_tasks//eval_interval
        results = [0]*final_num_tasks
        num_seeds = 25
        for i in range(num_seeds):
            print(f"*** STARTING SEED {i} for approach {approach} ***")
            if i == 0:
                difference = [0]*final_num_tasks #difference bw returns and target velocities

            rng = np.random.RandomState(i)
            torch.manual_seed(i)
            
            rewards, scores, targets = (test_single_approach(approach_fn, rng))

            for j in range(final_num_tasks):
                episode_return = sum(rewards[j])
                results[j] += episode_return/num_seeds

        pickle.dump([rewards, scores, targets, results], open(file, "wb"))


