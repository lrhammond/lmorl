# Various utility functions and constants

from itertools import product

# Constants
# UPDATE_EVERY = 4
# BUFFER_SIZE = int(1e6)
# BATCH_SIZE = 32
# EPSILON = 0.05
# LAMBDA_RL_2 = 0.05
# UPDATE_EVERY_EPS = 32
# SLACK = 0.04
# TOL = 1
# CONVERGENCE_LENGTH = 1000
# CONVERGENCE_DEVIATION = 0.04
# TOL2 = 1

##################################################

def make_sh_file(name):

    others = ['AC', 'RCPO', 'random', 'AproPO', 'VaR_AC', 'VaR_PG'] 
    ours = ['LA2C', 'seqLA2C', 'LA2C2nd', 'seqLA2C2nd', 'LPPO', 'seqLPPO', 'LPPO2nd', 'seqLPPO2nd']
    agent_names = others + ours
    robots = ['Point']
    tasks = ['Goal']
    difficulties = [1]
    episodes = [100000]
    iterations = [1,2,3,4,5]

    filename = '{}.sh'.format(name)
    with open(filename, 'w') as f:
        f.write('#!/bin/bash\n')
        for (a_n, r, t, d, e, i) in product(agent_names, robots, tasks, difficulties, episodes, iterations):
            f.write('sbatch experiment.sh {} {} {} {} {} {}\n'.format(a_n, r, t, d, e, i))

# make_sh_file('run_all')