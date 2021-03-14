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
                       hidden=16,
                       continuous=continuous)
        mode = 4

    elif agent_name=='tabular':
        agent = Tabular(action_size=action_size)
        mode = 1

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
        env = BalanceBotEnv(render=True)
    elif game == 'PuckEnv':
        env = PuckEnv()
    elif game == 'MountainCarSafe':
        env = MountainCarSafe()
    else:
        env = gym.make(game + '-v0')

    state = env.reset()
    state = np.expand_dims(state, axis=0)
    state = torch.tensor(state).float().to(device)

    env.render()

    for step in range(interacts):

        action = agent.act(state)

        if type(action)!=int:
            action = action.squeeze().cpu()
            if int_action:
                action = int(action)

        #print(action)
        if step % 10 == 0:
            print(action)

        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = torch.tensor(next_state).float().to(device)

        env.render()

        try:
            cost = info['cost']
        except:
            try:
                cost = info['constraint_costs'][0]
            except:
                cost = 0
                #print('cost exception, ', info)

        if mode==1:
            r = reward
        elif mode==2:
            r = reward-cost
        elif mode==3:
            r = [-cost, reward]
        elif mode==4:
            r = [reward, cost]
        elif mode==5:
            r = [reward, -cost]

        #agent.step(state, action, r, next_state, done)

        if done:
            step = 0
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state).float().to(device)
        else:
            state = next_state

    env.close()

##################################################


agent_name = 'LDQN'
game = 'CartSafe'
interacts = 10000
int_action=False

if game == 'CartSafe':
    i_s = 4
    a_s = 2
    hid = 32
    cont = False
    int_action=True
if game == 'GridNav':
    i_s = 625
    a_s = 4
    hid = 2048
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
    hid = 128
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

agent, mode = make_agent(agent_name, in_size=i_s, action_size=a_s, hidden=hid, network='DNN', continuous=cont, alt_lex=False)

agent.load_model('../results/{}/{}/{}-48403-3'.format(game, agent_name, agent_name))

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
