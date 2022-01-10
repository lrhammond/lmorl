import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

import matplotlib.pyplot as plt
import numpy as np
import os, collections
from tensorflow.python.summary.summary_iterator import summary_iterator

raise Exception("Not yet functional")

colours = {'AC': 'blue',
           'DQN': 'orange',
           'random': 'red',
           'LDQN': 'green',
           'LA2C': 'purple',
           'LPPO': 'cyan',
           'AproPO': 'pink',
           'RCPO': 'purple',
           'VaR_AC': 'grey',
           }

##           'LPPO2nd':'',
##           'seqLPPO':'',
##           'seqLPPO2nd':'',
##           'LA2C2nd':'',
##           'seqLA2C':'',
##           'seqLA2C2nd':'',

# algs = ['AC','DQN','LDQN','random','LA2C','LPPO']

# algs = ['LDQN','LPPO','LPPO2nd','seqLPPO','seqLPPO2nd']
# name = 'LPPO'

# algs = ['LDQN','LA2C','LA2C2nd','seqLA2C','seqLA2C2nd']
# name = 'LA2C'

algs = ['AC', 'DQN', 'random', 'LDQN', 'LA2C', 'LPPO', 'VaR_AC', 'AproPO', 'RCPO']
name = 'all'

base_path = "/run_archive/2/runs_tests/learn_tests/20220110-171827"

for mode in ['r', 'c']:

    max_steps_avg = None

    long_run = True

    step = 10000

    data = {}

    for alg in algs:

        print(alg)
        data[alg] = []

        # for run in range(runs):

        for _, _, files in os.walk(base_path):  # './{}/'.format(alg)):

            for file in files:

                if '.txt' or ".tfevents" in file:

                    print(file)

                    vals = []

                    if max_steps_avg:
                        last_few_vals = collections.deque(maxlen=max_steps_avg)
                    long_run_val = 0

                    # rewards = []
                    # costs = []

                    # long_run_reward = 0
                    # long_run_cost = 0

                    xs = []

                    if '.txt' in file:

                        with open('./{}/{}'.format(alg, file), 'r') as f:

                            for line in f:

                                l = line.strip().split(',')

                                if mode == 'r':
                                    xs.append(float(l[0]))
                                else:
                                    xs.append(float(l[1]))

                    elif '.tfevents' in file:

                        s_iter = summary_iterator(os.path.join(base_path, file))
                        for i, line in enumerate(s_iter):
                            if i > 1000: break
                            if mode == "r":
                                for val in line.summary.value:
                                    if "Reward" in val.tag:
                                        xs.append(val.simple_value)

                            elif mode == "c":
                                for val in line.summary.value:
                                    if "Cost" in val.tag:
                                        xs.append(val.simple_value)
                    print(xs)
                    for i, x in enumerate(xs):
                        if long_run:
                            if max_steps_avg:
                                long_run_val *= len(last_few_vals)
                                if len(last_few_vals) == max_steps_avg:
                                    long_run_val -= last_few_vals[0]
                                long_run_val += x
                                if len(last_few_vals) >= 1:
                                    long_run_val /= len(last_few_vals)

                                last_few_vals.append(x)
                            else:
                                long_run_val = (long_run_val * i + x) / (i + 1)

                        if long_run:
                            vals.append(long_run_val)
                        else:
                            vals.append(x)

                        # r = float(l[0])
                        # c = float(l[1])

                        # long_run_reward = (long_run_reward*i + r)/(i+1)
                        # long_run_cost = (long_run_cost*i + c)/(i+1)

                        # rewards.append(long_run_reward)
                        # costs.append(long_run_cost)

                        # rewards.append(r)
                        # costs.append(c)

                    # data[alg].append((rewards, costs))
                    data[alg].append(vals)

    for alg in algs:

        print()
        print(alg)
        try:
            min_length = min([len(d) for d in data[alg]])
            print(min_length)

            x = []
            y = []
            y_max = []
            y_min = []

            for i in range(min_length):

                if i % step == 0:
                    x.append(i)

                    m = np.mean([d[i] for d in data[alg]])
                    v = np.std([d[i] for d in data[alg]])

                    y.append(m)
                    y_max.append(m + v)
                    y_min.append(m - v)

            # y = [np.mean([d[i] for d in data[alg]]) for i in range(min_length)]
            # y_err = [np.std([d[i] for d in data[alg]]) for i in range(min_length)]

            # plt.errorbar(x, y, y_err, label = alg)
            plt.plot(x, y, label=alg, color=colours[alg])
            plt.fill_between(x, y_min, y_max, color=colours[alg], alpha=0.35)
        except:
            print('no data (?)')

    plt.xlabel('environment interacts')

    if mode == 'r':
        plt.ylabel('reward')
    else:
        plt.ylabel('cost')

    plt.legend(loc=1)

    # plt.xscale('log')

    t = "figs/" + ('reward' if mode == 'r' else 'cost') + '-{}'.format(name) + '.pdf'
    plt.savefig(t, format='pdf', dpi=400)
    plt.close()

# if mode == 'r':
#    plt.savefig('reward.png', format='png', dpi=400)
# else:
#    plt.savefig('cost.png', format='png', dpi=400)
# plt.close()

