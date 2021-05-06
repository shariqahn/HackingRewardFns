import pickle 
import matplotlib.pyplot as plt
import numpy as np

from test import num_tasks, eval_interval
# from test_env import max_returns


# print(pickle.load(open("sample.pkl",'rb')))
for approach in (
    # 'SingleTaskDDPG',
    'MultiTaskDDPG',
    # 'MultiTaskDDPGAugmentedOracle',
    # 'MultiTaskAugmentedOracle',
    # 'MultiTaskDQNOneQuery',
    # 'MultiTaskDQNTwoQuery',
    # 'MultiTaskDQN'
    ):
    print(approach)
# ('Random Policy', 'Single Task', 'Multitask', 'Single Task with Hacking', 'Multitask with Hacking'):
    # if approach == 'Random Policy':
    #     file = 'random.pkl'
    # elif approach == 'Single Task':
    #     file = 'single_task.pkl'
    # elif approach == 'Multitask':
    #     file = 'multitask.pkl'
    # elif approach == 'Single Task with Hacking':
    #     file = 'single_task_augmented.pkl'
    # elif approach == 'Multitask with Hacking':
    #     file = 'multitask_augmented.pkl'
    if approach == 'SingleTaskDQN':
        file = 'eval_single_dqn.pkl'
    elif approach == 'MultiTaskDQN':
        file = 'eval_multi_dqn.pkl'
    elif approach == 'SingleTaskAugmentedDQN':
        file = 'eval_single_augmented_dqn.pkl'
    elif approach == 'MultiTaskAugmentedDQN':
        file = 'eval_multi_augmented_dqn.pkl'
    elif approach == 'MultiTaskAugmentedOracle':
        file = 'eval_MultiTaskAugmentedOracle.pkl'
    elif approach == 'MultiTaskDQNOneQuery':
        file = 'eval_multi_augmented_dqn_1_query.pkl'
    elif approach == 'MultiTaskDQNTwoQuery':
        file = 'eval_multi_augmented_dqn_2_query.pkl'
    elif approach == 'SingleTaskDDPG':
        file = 'SingleTaskDDPG.pkl'
    elif approach == 'MultiTaskDDPG':
        file = 'MultiTaskDDPG.pkl'
    elif approach == 'MultiTaskDDPGAugmentedOracle':
        file = 'MultiTaskDDPGAugmentedOracle.pkl'

    rewards, scores, targets, results = pickle.load(open('results/'+file,'rb'))
    # num_eval_tasks = num_tasks//10
    num_eval_tasks = len(rewards)
    save_to = 'figures/DDPG/' + file[:-4] + '.png'
    
    # Data for plotting
    # x = eval_interval*np.arange(num_eval_tasks)
    results = []
    for r in rewards:
        results.append(sum(r))
    x = eval_interval*np.arange(len(results))
    fig, ax = plt.subplots()
    ax.plot(x,results)

    ax.set(xlabel='Number of Episodes', ylabel='Return',
           title=approach)
    # ax.set_ylim(ymax=0, ymin=-1e4)
    ax.grid()

    fig.savefig(save_to)

    # plot with max return reference line
    # plt.clf()
    # z = []
    # final_results = [] # returns from 25th test
    # for i in range(len(rewards)):
    #     target = targets[i]
    #     for j in range(len(rewards[i])):
    #         final_results.append(rewards[i][j])
    #         z.append(max_returns[target])

    # x = range(1,len(final_results)+1)
    # plt.plot(x,final_results, label = 'results')
    # plt.plot(x,z, label = 'max returns')

    # plt.xlabel('Number of Steps')
    # plt.ylabel('Reward')
    # plt.title(approach)
    # # plt.ylim(ymax=0, ymin=-1000000)

    # # ax.set_ylim(ymax=40,ymin=-300)
    # # ax.grid()
    # plt.legend()
    # plt.savefig(save_to[:-4] + '_reference.png')

    # # plot of scores
    # x = eval_interval*np.arange(num_eval_tasks)
    # z = scores

    # fig, ax = plt.subplots()
    # ax.plot(x,z)

    # ax.set(xlabel='Number of Episodes', ylabel='Policy Scores',
    #        title=approach)
    # # ax.set_ylim(ymax=0, ymin=-1000000)

    # ax.grid()

    # fig.savefig(save_to[:-4]+'_scores.png')

    # y = goals

    # fig, ax = plt.subplots()
    # ax.plot(x,y)

    # ax.set(xlabel='Number of Episodes', ylabel='Number of Times Goal Reached',
    #        title=approach)
    # ax.grid()

    # fig.savefig(save_to[:-4]+'_goals.png')

    # Plot state visits
    # fig, ax = plt.subplots()
    # ax.set(title=approach)
    # ax.axis('off')
    # plt.imshow(state_visits, vmin=0, cmap='jet')
    # plt.colorbar()
    # fig.savefig(save_to[:-4]+'_state_visits.png')

