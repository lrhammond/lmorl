# Main file for running experiments

import numpy as np
import torch
import random
import time
from itertools import count
import os
import sys
from multiprocessing import Pool
import safety_gym
import gym

##################################################

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

if torch.cuda.is_available():
    print("Using GPU!")
    device = torch.device("cuda")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("Using CPU!")
    device = torch.device("cpu")

##################################################

from utils import *
from learners_lexicographic import *
from learners_other import *

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
        agent = RCPO(action_size=action_size, constraint=1, in_size=in_size, 
                         network=network, hidden=hidden, continuous=continuous)
        mode = 4
        
    elif agent_name=='VaR_PG':
        agent = VaR_PG(action_size=action_size, alpha=1, beta=0.5, in_size=in_size, 
                           network=network, hidden=hidden, continuous=continuous)
        mode = 4
        
    elif agent_name=='VaR_AC':
        agent = VaR_AC(action_size=action_size, alpha=1, beta=0.5, in_size=in_size, 
                           network=network, hidden=hidden, continuous=continuous)
        mode = 4
        
    elif agent_name=='AproPO':
        constraints = [(10,35),(0,25)]
        agent = AproPO(action_size=action_size, 
                       in_size=in_size, 
                       constraints=constraints,
                       reward_size=2,
                       network=network, 
                       hidden=16,
                       continuous=continuous)
        mode = 4
        
    elif agent_name=='LAC':
        agent = LexActorCritic(in_size=in_size, action_size=action_size,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
            
    elif agent_name=='seqLAC':
        agent = LexActorCritic(in_size=in_size, action_size=action_size,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
            
    elif agent_name=='LNAC':
        agent = LexNaturalActorCritic(in_size=in_size, action_size=action_size,
                                      reward_size=2, network='DNN', hidden=hidden, sequential=False)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
            
    elif agent_name=='seqLNAC':
        agent = LexNaturalActorCritic(in_size=in_size, action_size=action_size,
                                      reward_size=2, network='DNN', hidden=hidden, sequential=True)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
            
    elif agent_name=='tabular':
        agent = Tabular(action_size=action_size)
        mode = 1
        
    elif agent_name=='random':
        agent = RandomAgent(action_size=action_size, continuous=continuous)
        mode = 1

    elif agent_name=='LA2C':
        agent = NewLexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
            
    elif agent_name=='seqLA2C':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LPPO':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
            
    elif agent_name=='seqLPPO':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LA2C2nd':
        agent = NewLexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
            
    elif agent_name=='seqLA2C2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name=='LPPO2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
            
    elif agent_name=='seqLPPO2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3
        
    else:
        print('invalid agent specification')
        assert False
        
    return agent, mode

##################################################

def run(agent, game, episodes, mode):
    
    env = gym.make('Safexp-{}-v0'.format(game))
    #env = gym.make('Safexp-PointGoal1-v0')

    performance_list = []
    safety_list = []

    for i in range(episodes):

        state = env.reset()
        state = np.expand_dims(state, axis=0)
        state = torch.tensor(state).float().to(device)
        
        done = False

        performance = 0
        safety = 0

        while not done:

            action = agent.act(state).cpu()
            
            next_state, reward, done, info = env.step(action)
            
            next_state = np.expand_dims(next_state, axis=0)
            next_state = torch.tensor(next_state).float().to(device)
            
            if mode==1:
                r = reward
            elif mode==2:
                r = reward-info['cost']
            elif mode==3:
                r = [-info['cost'], reward]
            elif mode==4:
                r = [reward, info['cost']]
            elif mode==5:
                r = [reward, -info['cost']]
                
            agent.step(state, action, r, next_state, done)
            state = next_state

            performance += reward
            safety += info['cost']

            #env.render()

        performance_list.append(performance)
        safety_list.append(safety)
        
    return performance_list, safety_list

##################################################

agent_name, robot, task, difficulty, episodes, iteration = 'AC', 'Point', 'Goal', 1, 1000, 1

# agent_name = sys.argv[1]
# robot = sys.argv[2]
# task = sys.argv[3]
# difficulty = sys.argv[4]
# episodes = int(sys.argv[5])
# iteration = int(sys.argv[6])

directory = robot + task + difficulty

os.makedirs('./results/{}/{}'.format(directory, agent_name), exist_ok=True)
process_id = str(time.time())[-5:]

##################################################

seed = iteration
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

##################################################

def evaluate(i):

    agent, mode = make_agent(agent_name, in_size=60, action_size=2, hidden=256, network='DNN', continuous=True, alt_lex=False)
    game = robot + task + difficulty

    print('Running agent')
    reward, cost = run(agent, game, episodes, mode)
    print('Saving data...')
    
    with open('./results/{}/{}/{}.txt'.format(directory, agent_name, i), 'a') as f:
        for r,c in zip(reward, cost):
            f.write('{},{}\n'.format(r,c))

evaluate(iteration)
#p = Pool(iterations)
#p.map(evaluate, list(range(iterations)))