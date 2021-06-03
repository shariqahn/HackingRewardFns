import pickle 
import matplotlib.pyplot as plt
import numpy as np

from test import num_tasks, eval_interval
# from test_env import max_returns


# print(pickle.load(open("sample.pkl",'rb')))
for approach in (
    'SingleTaskDQN',
    'SingleTaskAugmentedDQN',
    # 'MultiTaskAugmentedOracle',
    # 'MultiTaskDQNOneQuery',
    # 'MultiTaskDQNTwoQuery',
    # 'MultiTaskDQN',
    # 'SingleTaskDDPG',
    # 'MultiTaskDDPG',
    # 'MultiTaskDDPGAugmentedOracle',
    # 'MultiTaskDDPGQuery',
    # 'MultiTaskDDPGAutoQuery',
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
        file = '_eval_single_dqn.pkl'
    elif approach == 'MultiTaskDQN':
        file = '_eval_multi_dqn.pkl'
    elif approach == 'SingleTaskAugmentedDQN':
        file = '_eval_single_augmented_dqn.pkl'
    elif approach == 'MultiTaskAugmentedOracle':
        file = '_eval_MultiTaskAugmentedOracle.pkl'
    elif approach == 'MultiTaskDQNOneQuery':
        file = '_eval_multi_augmented_dqn_1_query.pkl'
    elif approach == 'MultiTaskDQNTwoQuery':
        file = '_eval_multi_augmented_dqn_2_query.pkl'
    elif approach == 'SingleTaskDDPG':
        file = 'SingleTaskDDPG.pkl'
    elif approach == 'MultiTaskDDPG':
        file = 'MultiTaskDDPG.pkl'
    elif approach == 'MultiTaskDDPGAugmentedOracle':
        file = 'MultiTaskDDPGAugmentedOracle.pkl'
    elif approach == 'MultiTaskDDPGQuery':
        file = 'MultiTaskDDPGQuery.pkl'
    elif approach == 'MultiTaskDDPGAutoQuery':
        file = 'MultiTaskDDPGAutoQuery.pkl'

    rewards, positions, targets, results = pickle.load(open('results/'+file,'rb'))
    num_eval_tasks = len(rewards)
    # save_to = 'figures/DDPG/' + file[:-4] + '.png'
    save_to = 'figures/DQN/' + file[:-4] + '.png'
    
    # Data for plotting
    # x = eval_interval*np.arange(num_eval_tasks)

    x = eval_interval*np.arange(len(results))
    fig, ax = plt.subplots()
    ax.plot(x,results)

    # std = np.std(rewards, axis=0)
    # print(std)
    # print(len(results), len(rewards[0]), len(std))

    # ax.fill_between(x, results-std, results+std, alpha=0.25)

    ax.set(xlabel='Number of Episodes', ylabel='Return',
           title=approach)
    # ax.set_ylim(ymax=6200, ymin=-1250)
    # ax.set_ylim(ymax=0, ymin=-10000)
    ax.grid()

    fig.savefig(save_to)


    # plots states across last 2 episodes for each seed in separate line
    # fig, ax = plt.subplots()
    # for seed in range(5):
    #     for i in range(-2,0,1):
    #         episode = positions[seed][i]
    #         # print(targets[i])
    #         if targets[seed][i] == 1.0:
    #             ax.plot(range(len(episode)), episode, label='direction = right', color='b')
    #         else:
    #             ax.plot(range(len(episode)), episode, label='direction = left', color='r')
    # ax.set(xlabel='Number of Steps', ylabel='Position',
    #        title=approach + ' Episode Behavior')
    # ax.set_ylim(ymax=.6, ymin=-.6)
    # ax.grid()
    # # ax.legend()

    # fig.savefig(save_to[:-4] + '_episodes.png')

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

    # plot of scores
    x = eval_interval*np.arange(num_eval_tasks)
    # z = scores
    z = positions

    fig, ax = plt.subplots()
    ax.plot(x,z)

    ax.set(xlabel='Number of Episodes', ylabel='Policy Scores',
           title=approach)
    ax.set_ylim(ymax=2, ymin=-1)

    ax.grid()

    fig.savefig(save_to[:-4]+'_scores.png')

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

