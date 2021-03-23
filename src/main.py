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

# from open_safety.envs.balance_bot_env import BalanceBotEnv
# from open_safety.envs.puck_env import PuckEnv

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

def make_agent(agent_name, in_size=60, action_size=4, hidden=256, network='DNN', continuous=False):

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
        agent = RCPO(action_size=action_size, constraint=5.0, in_size=in_size,
                         network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name=='VaR_PG':
        agent = VaR_PG(action_size=action_size, alpha=5, beta=0.95, in_size=in_size,
                           network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name=='VaR_AC':
        agent = VaR_AC(action_size=action_size, alpha=5, beta=0.95, in_size=in_size,
                           network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name=='AproPO':
        constraints = [(90.0, 100.0),(0.0, 5.0)]
        agent = AproPO(action_size=action_size,
                       in_size=in_size,
                       constraints=constraints,
                       reward_size=2,
                       network=network,
                       hidden=hidden,
                       continuous=continuous)
        mode = 4

##    elif agent_name=='LAC':
##        agent = LexActorCritic(in_size=in_size, action_size=action_size,
##                               reward_size=2, network='DNN', hidden=hidden, sequential=False)
##        if prioritise_performance_over_safety:
##            mode = 5
##        else:
##            mode = 3
##
##    elif agent_name=='seqLAC':
##        agent = LexActorCritic(in_size=in_size, action_size=action_size,
##                               reward_size=2, network='DNN', hidden=hidden, sequential=True)
##        if prioritise_performance_over_safety:
##            mode = 5
##        else:
##            mode = 3
##
##    elif agent_name=='LNAC':
##        agent = LexNaturalActorCritic(in_size=in_size, action_size=action_size,
##                                      reward_size=2, network='DNN', hidden=hidden, sequential=False)
##        if prioritise_performance_over_safety:
##            mode = 5
##        else:
##            mode = 3
##
##    elif agent_name=='seqLNAC':
##        agent = LexNaturalActorCritic(in_size=in_size, action_size=action_size,
##                                      reward_size=2, network='DNN', hidden=hidden, sequential=True)
##        if prioritise_performance_over_safety:
##            mode = 5
##        else:
##            mode = 3
##
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
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
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

def run(agent, env, interacts, max_ep_length, mode, save_location, int_action=False):

    state = env.reset()
    state = np.expand_dims(state, axis=0)
    state = torch.tensor(state).float().to(device)

    filename = './{}/{}/{}/{}-{}.txt'.format(save_location, game, agent_name, agent_name, process_id)

    step = 0

    for i in range(interacts):

        action = agent.act(state)
        step += 1

        if int_action:
            action = int(action)
        if type(action)!=int:
            action = action.squeeze().cpu().float()

        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = torch.tensor(next_state).float().to(device)

        try:
            cost = info['cost']
        except:
            try:
                cost = info['constraint_costs'][0]
            except:
                cost = 0

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

        #time.sleep(0.0001)
        agent.step(state, action, r, next_state, done)

        if done or (step >= max_ep_length):
            step = 0
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state).float().to(device)
        else:
            state = next_state

        with open(filename, 'a') as f:
            f.write('{},{}\n'.format(reward, cost))

        #if i % 1000000 == 0:
        #    agent.save_model('./{}/{}/{}/{}-{}-{}'.format(save_location, game, agent_name, agent_name, process_id, i))

    agent.save_model('./{}/{}/{}/{}-{}'.format(save_location, game, agent_name, agent_name, process_id))


def run_episodic(agent, env, episodes, max_ep_length, mode, save_location, int_action=False):

    filename = './{}/{}/{}/{}-{}.txt'.format(save_location, game, agent_name, agent_name, process_id)

    for i in range(episodes):

        state = env.reset()
        state = np.expand_dims(state, axis=0)
        state = torch.tensor(state).float().to(device)

        cum_reward = 0
        cum_cost = 0

        for j in range(max_ep_length):

            action = agent.act(state)

            if int_action:
                action = int(action)
            if type(action)!=int:
                action = action.squeeze().cpu().float()

            next_state, reward, done, info = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            next_state = torch.tensor(next_state).float().to(device)

            try:
                cost = info['cost']
            except:
                try:
                    cost = info['constraint_costs'][0]
                except:
                    cost = 0

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

            cum_reward += reward
            cum_cost += cost

            agent.step(state, action, r, next_state, done)

            if done:
                break
            
            state = next_state

        with open(filename, 'a') as f:
            f.write('{},{}\n'.format(cum_reward, cum_cost))

        #if i % 5000 == 0:
        #    agent.save_model('./{}/{}/{}/{}-{}-{}'.format(save_location, game, agent_name, agent_name, process_id, i))

    agent.save_model('./{}/{}/{}/{}-{}'.format(save_location, game, agent_name, agent_name, process_id))



##################################################

agent_name = sys.argv[1]
game = sys.argv[2]
interacts = int(sys.argv[3])

save_location = 'results'

os.makedirs('./{}/{}/{}'.format(save_location, game, agent_name), exist_ok=True)
process_id = str(time.time())[-5:]

seed = int(process_id)

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

int_action = False
cont = False
max_ep_length = 500

##################################################

if game == 'CartSafe':
    hid = 8
    int_action=True
    max_ep_length = 300

elif game == 'GridNav':
    hid = 128
    int_action = True
    max_ep_length = 50

elif game == 'MountainCarContinuousSafe':
    hid = 32
    cont = True
    max_ep_length = 200

elif game == 'MountainCar':
    hid = 32
    int_action = True
    max_ep_length = 200

elif game == 'MountainCarSafe':
    hid = 32
    int_action = True
    max_ep_length = 300

elif game == 'BalanceBotEnv':
    hid = 32
    max_ep_length = 300
    cont = True

elif game == 'Gaussian':
    hid = 8
    cont = True
    max_ep_length = 200

else:
    print('Invalid environment specification \"{}\"'.format(game))

##################################################

if game == 'BalanceBotEnv':
    env = BalanceBot()
    action_size = 2
    in_size = 32
    
elif game == 'MountainCarSafe':
    env = MountainCarSafe()
    action_size = 3
    in_size = 2

#elif game == 'GridNav':
#    env = GridNav()
#    action_size = 4
#    in_size = 625
    
elif game == 'Gaussian':
    env = Simple1DEnv()
    action_size = 1
    in_size = 1

elif game == 'CartSafe':
    env = CartSafe()
    action_size = 2
    in_size = 4
    
else:
    env = gym.make(game + '-v0')
    try:
        action_size = env.action_space.n
    except:
        action_size = len(env.action_space.high)
    in_size = len(env.observation_space.high)

##################################################
    
agent, mode = make_agent(agent_name, in_size, action_size, hid, 'DNN', cont)

run_episodic(agent, env, interacts, max_ep_length, mode, save_location, int_action)










