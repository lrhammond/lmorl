# Main file for running experiments

import numpy as np

import random
import collections
import time
from itertools import count
import os, sys, math
from multiprocessing import Pool
import cv2

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

def make_agent(agent_name, in_size=60, action_size=4, hidden=256, network='DNN', is_action_cont=False, alt_lex=False):

    prioritise_performance_over_safety = False

    if agent_name=='AC':
        agent = ActorCritic(action_size=action_size, in_size=in_size,
                                network=network, hidden=hidden, is_action_cont=is_action_cont)
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
                         network=network, hidden=hidden, is_action_cont=is_action_cont)
        mode = 4

    elif agent_name=='VaR_PG':
        agent = VaR_PG(action_size=action_size, alpha=1, beta=0.95, in_size=in_size,
                           network=network, hidden=hidden, is_action_cont=is_action_cont)
        mode = 4

    elif agent_name=='VaR_AC':
        agent = VaR_AC(action_size=action_size, alpha=1, beta=0.95, in_size=in_size,
                           network=network, hidden=hidden, is_action_cont=is_action_cont)
        mode = 4

    elif agent_name=='AproPO':
        constraints = [(0.3, 0.5),(0.0, 0.1)]
        agent = AproPO(action_size=action_size,
                       in_size=in_size,
                       constraints=constraints,
                       reward_size=2,
                       network=network,
                       hidden=16,
                       is_action_cont=is_action_cont)
        mode = 4

    elif agent_name=='tabular':
        agent = Tabular(action_size=action_size)
        mode = 1

    elif agent_name=='random':
        agent = RandomAgent(action_size=action_size, is_action_cont=is_action_cont)
        mode = 1

    elif agent_name=='LA2C':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, is_action_cont=is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='seqLA2C':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, is_action_cont=is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LPPO':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, is_action_cont=is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='seqLPPO':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, is_action_cont=is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LA2C2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, is_action_cont=is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='seqLA2C2nd':
        agent = ActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, is_action_cont=is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LPPO2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, is_action_cont=is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='seqLPPO2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, is_action_cont=is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LexTabular':
        agent = LexTabular(action_size=action_size)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='invLexTabular':
        agent = LexTabular(action_size=action_size)
        mode = 5

    else:
        print('invalid agent specification')
        assert False

    return agent, mode

##################################################


##################################################

agent_name = 'AC'

agent, mode = make_agent(agent_name, 625, 4, 32, 'DNN', False, alt_lex=False)

env = gym.make('GridNav-v0')

import matplotlib.pyplot as plt

for _,_, files in os.walk('../results/GridNav/{}/'.format(agent_name)):
    for file in filter(lambda file: '.txt' in file, files):

        print(file)
        
        agent.load_model('../results/GridNav/{}/'.format(agent_name)+file[:-4])

        up    = []
        down  = []
        right = []
        left  = []

        for s in range(0, env.gridsize**2):
                
            state = np.zeros((1,env.gridsize**2))
            state[0,s] = 1
            state = torch.tensor(state).float()
            
            lst = [agent.act(state) for _ in range(25)]
            action = int(max(set(lst), key=lst.count))

            if action == 2:
                down.append(s)
            elif action == 1:
                left.append(s)
            elif action == 0:
                up.append(s)
            elif action == 3:
                right.append(s)

        img = env.state_to_img()
        #img = cv2.resize(img, (env.display_size,env.display_size), interpolation=cv2.INTER_NEAREST)

        fig = plt.figure(0)
        plt.clf()
        plt.imshow(img)
        fig.canvas.draw()

        plt.plot([s % env.gridsize for s in up], [int(s/env.gridsize) for s in up], '^')
        plt.plot([s % env.gridsize for s in right], [int(s/env.gridsize) for s in right], '>')
        plt.plot([s % env.gridsize for s in left], [int(s/env.gridsize) for s in left], '<')
        plt.plot([s % env.gridsize for s in down], [int(s/env.gridsize) for s in down], 'v')

        #plt.plot([624 % env.gridsize],[int(624/env.gridsize)],'x')
        
        plt.savefig('../results/GridNav/' + file[:-4] + '.png')
        plt.close()

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

#run(agent, game, mode, int_action)


# AC 64436
# rand 43575
# RCPO 28656
