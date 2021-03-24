# Main file for running experiments

import numpy as np

import random
import collections
import time
from itertools import count
import os, sys, math
from multiprocessing import Pool

import gym
#import safety_gym
import gym_safety
import open_safety

from open_safety.envs.balance_bot_env import BalanceBotEnv
from open_safety.envs.puck_env import PuckEnv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
from torch.autograd import Variable

##################################################

from utils import *
from networks import *
from learners_lexicographic import *
from learners_other import *
from envs import *

##################################################


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

##if torch.cuda.is_available():
##    print("Using GPU!")
##    device = torch.device("cuda")
##    torch.set_default_tensor_type('torch.cuda.FloatTensor')
##else:
##    print("Using CPU!")
##    device = torch.device("cpu")

device = torch.device("cpu")

##################################################

def make_agent(agent_name, in_size=60, action_size=4, hidden=256, network='DNN', continuous=False, alt_lex=False):

    prioritise_performance_over_safety = False

    if agent_name=='AC':
        agent = ActorCritic(action_size=action_size, in_size=in_size,
                                network=network, hidden=hidden, continuous=continuous)
        mode = 1

    elif agent_name=='DQN':
        agent = DQN(action_size=action_size, in_size=in_size,
                        network=network, hidden=hidden)
        mode = 1

    elif agent_name=='LDQN':
        agent = LexDQN(action_size=action_size, in_size=in_size, reward_size=2,
                           network=network, hidden=hidden)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='RCPO':
        agent = RCPO(action_size=action_size, constraint=0.1, in_size=in_size,
                         network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name=='VaR_PG':
        agent = VaR_PG(action_size=action_size, alpha=1, beta=0.95, in_size=in_size,
                           network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name=='VaR_AC':
        agent = VaR_AC(action_size=action_size, alpha=1, beta=0.95, in_size=in_size,
                           network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name=='AproPO':
        constraints = [(0.3, 0.5),(0.0, 0.1)]
        agent = AproPO(action_size=action_size,
                       in_size=in_size,
                       constraints=constraints,
                       reward_size=2,
                       network=network,
                       hidden=hidden,
                       continuous=continuous)
        mode = 4

    elif agent_name=='tabular':
        agent = Tabular(action_size=action_size)
        mode = 1

    elif agent_name=='LexTabular':
        agent = LexTabular(action_size=action_size)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='invLexTabular':
        agent = LexTabular(action_size=action_size)
        mode = 5

    elif agent_name=='random':
        agent = RandomAgent(action_size=action_size, continuous=continuous)
        mode = 1

    elif agent_name=='LA2C':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='seqLA2C':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LPPO':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='seqLPPO':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LA2C2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='seqLA2C2nd':
        agent = ActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LPPO2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='seqLPPO2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    else:
        print('invalid agent specification')
        assert False

    return agent, mode

##################################################


def run(agent, game, mode, int_action=False):

    if game == 'BalanceBotEnv':
        if render:
            env = BalanceBot(render=True)
        else:
            env = BalanceBot(render=False)
    elif game == 'PuckEnv':
        env = PuckEnv()
    elif game == 'MountainCarSafe':
        env = MountainCarSafe()
    else:
        env = gym.make(game + '-v0')

    state = env.reset()
    state = np.expand_dims(state, axis=0)
    state = torch.tensor(state).float().to(device)

    if render:
        env.render()
    i = 0

    cumulative_cost = 0
    cumulative_reward = 0

    cumulative_costs = []
    cumulative_rewards = []

    for step in range(interacts):

        action = agent.act(state)
        i += 1

        #action = int(action)
        if type(action)!=int:
            action = action.squeeze().cpu()
            if int_action:
                action = int(action)

        #print(action)
##        if step % 10 == 0:
##            print()
##            print(action)
##            for _ in range(5):
##                print(float(agent.act(state)))
##            print()

        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = torch.tensor(next_state).float().to(device)

        if render:
            env.render()

        try:
            cost = info['cost']
        except:
            try:
                cost = info['constraint_costs'][0]
            except:
                cost = 0
                
        cumulative_cost += cost
        cumulative_reward += reward

        if done or i >= max_step:

            #print(state)
            #print(next_state)
            #print()

            print(i)

            #print('cumulative reward: {}'.format(cumulative_reward))
            #print('cumulative cost: {}'.format(cumulative_cost))

            cumulative_costs.append(cumulative_cost)
            cumulative_rewards.append(cumulative_reward)

            if len(cumulative_costs) % 10 == 0:
                print(len(cumulative_costs))

            cumulative_cost = 0
            cumulative_reward = 0

            i = 0
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state).float().to(device)
        else:
            state = next_state

    print()
    print('episodes: {}'.format(len(cumulative_rewards)))
    print('mean reward: {}'.format(sum(cumulative_rewards)/len(cumulative_rewards)))
    print('mean cost: {}'.format(sum(cumulative_costs)/len(cumulative_costs)))

    env.close()

##################################################

agent_name = 'LPPO'
game = 'BalanceBotEnv'
version = 0

render = True

interacts = 100000
max_step = 300
int_action=False

'CartSafe:'

'''
LA2C

episodes: 335
mean reward: 66.05724363935079
mean cost: 16.567164179104477

episodes: 333
mean reward: 58.30409285732141
mean cost: 1.0690690690690692

episodes: 333
mean reward: 92.9420071928909
mean cost: 3.960960960960961

episodes: 335
mean reward: 162.85607315291267
mean cost: 19.507462686567163

episodes: 333
mean reward: 95.94924971441544
mean cost: 14.327327327327327

episodes: 333
mean reward: 87.02220382689487
mean cost: 4.333333333333333

episodes: 333
mean reward: 100.42209297644564
mean cost: 4.054054054054054

episodes: 340
mean reward: 56.39526542248087
mean cost: 46.30882352941177

episodes: 333
mean reward: 16.861143846970414
mean cost: 0.5525525525525525

episodes: 592
mean reward: 35.33088809537506
mean cost: 59.876689189189186
'''


'''
LPPO

episodes: 335
mean reward: 74.61730871869484
mean cost: 37.602985074626865

episodes: 333
mean reward: 49.750443329450846
mean cost: 1.0990990990990992

episodes: 333
mean reward: 75.15519426946759
mean cost: 1.3963963963963963

episodes: 333
mean reward: 83.57504857039663
mean cost: 3.4504504504504503

episodes: 333
mean reward: 72.01744290698342
mean cost: 4.666666666666667

episodes: 333
mean reward: 95.90507410980115
mean cost: 2.6876876876876876

episodes: 333
mean reward: 51.65555492487361
mean cost: 2.018018018018018

episodes: 339
mean reward: 109.95227535271151
mean cost: 44.05309734513274

episodes: 333
mean reward: 51.23009639529476
mean cost: 0.3843843843843844

episodes: 333
mean reward: 52.8936635367478
mean cost: 2.285285285285285
'''



'???:'

'''
0
episodes: 1045
mean reward: 57.63714434476072
mean cost: 63.311004784689

1
episodes: 1501
mean reward: 99.61017488580045
mean cost: 30.59493670886076

2
episodes: 1002
mean reward: 109.71392396219872
mean cost: 7.164670658682635

3
episodes: 1000
mean reward: 44.41469999692886
mean cost: 0.0

4
episodes: 1000
mean reward: 2.6676060450390096
mean cost: 0.0
'''

if game == 'CartSafe':
    i_s = 4
    a_s = 2
    hid = 8
    cont = False
    int_action=True
if game == 'GridNav':
    i_s = 625
    a_s = 4
    hid = 128
    cont = False
    int_action = True
if game == 'MountainCarContinuousSafe':
    i_s = 2
    a_s = 2
    hid = 16
    cont = True
if game == 'PuckEnv':
    i_s = 18
    a_s = 2
    hid = 128
    cont = True
if game == 'BalanceBotEnv':
    i_s = 32
    a_s = 2
    hid = 32
    cont = True
if game == 'MountainCar':
    i_s = 2
    a_s = 3
    hid = 32
    cont = False
    int_action=True
if game == 'MountainCarSafe':
    i_s = 2
    a_s = 3
    hid = 32
    cont = False
    int_action=True

agent, mode = make_agent(agent_name, i_s, a_s, hid, 'DNN', cont, alt_lex=False)

i = 0
has_loaded = False
for _,_, files in os.walk('../results/{}/{}/'.format(game, agent_name)):
    for file in filter(lambda file: '.txt' in file, files):
        if i == version:
            print('../results/{}/{}/{}'.format(game, agent_name,file[:-4]))
            agent.load_model('../results/{}/{}/{}-3000000'.format(game, agent_name,file[:-4]))
            has_loaded=True
            break
        i += 1
if not has_loaded:
    print('using random initialisation!')

# t = '1616587665.7254992'
# agent.actor.load_state_dict(torch.load('/Users/lewishammond/Repositories/code/lmorl/actor_nan_error_{}.pt'.format(t)))
# agent.critic.load_state_dict(torch.load('/Users/lewishammond/Repositories/code/lmorl/critic_nan_error_{}.pt'.format(t)))
# with open('nan_error_data_{}.pickle'.format(t), "rb") as input_file:
#     p = pickle.load(input_file) 

import matplotlib.pyplot as plt
##
##x = []
##y = []
##colours = []
##
##for pos in np.arange(-1.2, 0.6, 0.03):
##    for vel in np.arange(-0.07, 0.07, 0.003):
##        
##        x.append(pos)
##        y.append(vel)
##
##        s = np.asarray([pos, vel])
##        s = np.expand_dims(s, axis=0)
##        s = torch.tensor(s).float().to(device)
##        
##        lst = [agent.act(s) for _ in range(10)]
##        action = max(set(lst), key=lst.count)
##        if action == 0:
##            colours.append('r')
##        if action == 1:
##            colours.append('y')
##        if action == 2:
##            colours.append('g')
##
##plt.scatter(x,y,c=colours)
##plt.show()

##for param in agent.critic.parameters():
##    print(param.shape)
##    #print(param.data)
##    print(True in torch.isnan(param.data))
##
##print()
##print('#################')
##print()
##
##for param in agent.actor.parameters():
##    print(param.shape)
##    #print(param.data)
##    print(True in torch.isnan(param.data))
##
##print()
##print('#################')
##print()

run(agent, game, mode, int_action)


# AC 64436
# rand 43575
# RCPO 28656
