import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os, collections
from pathlib import Path

mpl.rcParams['agg.path.chunksize'] = 10000

COLOURS = {
    'AC': 'blue',
    'DQN': 'orange',
    'random': 'red',
    'LDQN': 'green',
    'LA2C': 'purple',
    'LPPO': 'cyan',
    'AproPO': 'pink',
    'RCPO': 'purple',
    'VaR_AC': 'grey',
    'LPPO2nd': 'coral',
    'seqLPPO': 'goldenrod',
    'seqLPPO2nd': 'yellowgreen',
    'LA2C2nd': 'lightseagreen',
    'seqLA2C': 'steelblue',
    'seqLA2C2nd': 'mediumpurple'
}

ALGS = list(COLOURS.keys())
# BASE_DIR = os.path.join(Path.home(), "lmorl/data/txts/")
BASE_DIR = "/home/apeex/lmorl/data/txts/recreating_fig2_CartSafe_long"


def main():
    reward_data = get_data_dict_for_alg_comparison()
    graph_from_dict(reward_data, "reward", "reward_example")

    cost_data = get_data_dict_for_alg_comparison(reward_or_cost="c")
    graph_from_dict(cost_data, "cost", "cost_example")


def get_data_dict_for_alg_comparison(reward_or_cost: str = "r",
                                     max_steps_avg: int = None,
                                     long_run: bool = True,
                                     ):
    data = {}
    assert (reward_or_cost in ["r", "c"])

    for alg in ALGS:

        print(alg)
        data[alg] = []
        for dr, _, files in os.walk(BASE_DIR):

            for file in files:
                file_path = os.path.join(dr, file)

                if '.txt' in file_path and alg in file_path:

                    vals = []

                    if max_steps_avg:
                        last_few_vals = collections.deque(maxlen=max_steps_avg)
                    long_run_val = 0

                    with open(file_path, 'r') as f:

                        for i, line in enumerate(f):

                            l = line.strip().split(',')

                            if reward_or_cost == 'r':
                                x = float(l[0])
                            else:
                                x = float(l[1])

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

                        data[alg].append(vals)
    return data


def graph_from_dict(data, y_label: str, graph_name: str, step: int = 10000):
    for alg in data.keys():

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

            print(y_min, y_max)
            plt.plot(x, y, label=alg, color=COLOURS[alg])
            plt.fill_between(x, y_min, y_max, color=COLOURS[alg], alpha=0.35)
        except:
            print('no data (?)')
    plt.xlabel('environment interacts')
    plt.ylabel(y_label)
    plt.legend(loc=1)
    t = graph_name + '.pdf'
    plt.savefig(t, format='pdf', dpi=400)
    plt.close()

if __name__=="__main__":
    main()