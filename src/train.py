# MAIN ORDER:
# TODO update "continuous" var name
# TODO get  learnng
# TODO get AC learnng
# TODO get LDQN learnng
# TODO get LA2C learnng
# TODO get LPPO learnng
# TODO get AproPO learnng
# TODO get RCPO learnng
# TODO get VaR_AC learnng
# TODO get Joar's cartsafe working (+ learning) (ask for hyperparams)
# BUFFER_SIZE = int(1e5)
# BATCH_SIZE = 8
# UPDATE_EVERY = 8
# UPDATE_EVERY_EPS = 1
#
# EPSILON = 0.05
# SLACK = 0.04
# LAMBDA_LR_2 = 0.05
#
# LR = 1e-3
#
# update_steps = 10
#
# network size: 4 x 32 x 32 x 2
# max_ep_length = 300
# TODO Cartsafe working on cluster
# TODO get one OpenAI's safety environment working on cluster (point, goal, lvl 1)
# TODO look for existing hyper param + models for safety-gym
# TODO upgrade to ARC or some kind of cluster
# TODO get decent learning for safety environment (point+car, -- ,lvl1+2)
# TODO maybe try 3D?

# SIDE-QUESTS
# TODO refactor for readibility
# TODO move (appropriate) locally defined variables to arg parser

# Main file for running experiments
import datetime
import inspect
# import numpy as np

import random
# import collections
import time
# from itertools import count
import os# , sys, math
# from multiprocessing import Pool
import argparse
from src.envs import get_env_and_params, env_names
# import copy
from tqdm import tqdm

import gym

gym.logger.set_level(40)
# import safety_gym
# import gym_safety
# import open_safety

# from open_safety.envs.balance_bot_env import BalanceBotEnv
# from open_safety.envs.puck_env import PuckEnv

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms as T
# from torch.distributions import Categorical
# from torch.autograd import Variable
# from tensorboard.summary import Writer
from torch.utils.tensorboard import SummaryWriter

##################################################
from src.make_agent import make_agent, agent_names
# from utils import *
# from networks import *
# from learners_lexicographic import *
# from learners_other import *
from src.envs import *


##################################################

class TrainingParameters():
    def __init__(self,
                 agent_name: str,
                 env_name: str,
                 num_episodes: int,
                 save_location: str = "results",
                 network: str = "DNN",
                 test_group_label: str = None
                 ):
        assert (agent_name in agent_names)
        assert (env_name in env_names)
        assert (network in ["CNN", "DNN"])
        self.agent_name = agent_name
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.save_location = save_location
        self.network = network
        self.test_group_label = test_group_label

    def render(self):
        print()
        print("===Training Parameters===")
        for atr_name, atr in inspect.getmembers(self):
            if not atr_name.startswith("_") and not inspect.ismethod(atr):
                print(f"{atr_name}: {str(atr)}")
        print("===  ===  =====  ===  ===")
        print()

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

# device = torch.device("cpu")


def run_episodic(agent, env, num_episodes, args, max_ep_length, mode, save_location, file_pref,
                 device, int_action=False, tb_log_path=None, agent_name=None, env_name=None, show_ep_prog_bar=False):
    internal_train_log = []
    if env_name == "Bandit":
        action_seq = []
        dt_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        bandit_extra_dir = os.path.join("runs_bandit_extra", agent_name, dt_string)
        bandit_extra_writer = SummaryWriter(log_dir=bandit_extra_dir)
    if tb_log_path is None:
        writer = SummaryWriter(comment="_" + agent_name)
    else:
        writer = SummaryWriter(log_dir=tb_log_path)

    if show_ep_prog_bar:
        episode_iter = tqdm(range(num_episodes))
    else:
        episode_iter = range(num_episodes)
    total_interacts = 0
    for i in episode_iter:
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        state = torch.tensor(state).float().to(device)

        cum_reward = 0
        cum_cost = 0

        for j in range(max_ep_length):
            action = agent.act(state)

            if env_name=="Bandit":
                action_seq.append(action)

            total_interacts += 1

            if int_action:
                action = int(action)
            if type(action) != int:
                action = action.squeeze().cpu().float()

            next_state, reward, done, info = env.step(action)
            # print(i, action, reward)
            next_state = np.expand_dims(next_state, axis=0)
            next_state = torch.tensor(next_state).float().to(device)

            try:
                cost = info['cost']
            except:
                try:
                    cost = info['constraint_costs'][0]
                except:
                    cost = 0

            if mode == 1:
                r = reward
            elif mode == 2:
                r = reward - cost
            elif mode == 3:
                r = [-cost, reward]
            elif mode == 4:
                r = [reward, cost]
            elif mode == 5:
                r = [reward, -cost]

            cum_reward += reward
            cum_cost += cost

            agent.step(state, action, r, next_state, done)

            if done:
                break

            state = next_state

        with open(file_pref + '.txt', 'a') as f:
            f.write(f'| Episode: {i:03d} |  '
                    f'| Cumulative reward: {cum_reward:>8.3f} |  '
                    f'| Cumulative cost: {cum_cost:>8.3f} |\n')

        # if i % 5000 == 0:
        #    agent.save_model('./{}/{}/{}/{}-{}-{}'.format(save_location, game, agent_name, agent_name, process_id, i))

        internal_train_log.append((i, cum_reward, cum_cost))

        writer.add_scalar(f"{env_name}/Reward", cum_reward, i)

        writer.add_scalar(f"{env_name}/Cost", cum_cost, i)

        from src.learners_other import Tabular
        if total_interacts%100 == 0 and env_name == "Bandit" and isinstance(agent, Tabular):
            for state in agent.Q.keys():
                for action in agent.Q[state].keys():
                    bandit_extra_writer.add_scalar(f"{env_name}/Reward/train/{state}-{action}",  agent.Q[state][action], i)

    if env_name == "Bandit":
        for a_name in [0,1,2]:
            resolution = 100
            split = np.array([1 if a == a_name else 0 for a in action_seq])
            running_mean_1 = np.array([split[0:i+1].mean() for i in range(resolution-1)])
            running_mean_2 = np.convolve(split, np.ones(resolution)/resolution, mode='valid')
            running_mean = np.concatenate((running_mean_1, running_mean_2), axis=0)
            for i, m in enumerate(running_mean):
                bandit_extra_writer.add_scalar(f"{env_name}/Rolling {resolution} freq. of action {a_name}", m, i)

    agent.save_model(file_pref)
    writer.flush()
    return internal_train_log


##################################################

def get_train_params_from_args():
    # TODO change default running command in README

    # agent_name, game, interacts = 'LPPO', 'CartSafe', 10000
    parser = argparse.ArgumentParser(description="Run lexico experiments")

    parser.add_argument("--agent_name", type=str, default="tabular", choices=agent_names,
                        help="The name of the type of agent (e.g AC, DQN, LDQN)")

    parser.add_argument("--env_name", type=str, default="Gaussian", choices=env_names,
                        help="The name of the game to train on e.g. 'MountainCarSafe', 'Gaussian', 'CartSafe':")

    parser.add_argument("--num_episodes", type=int, default=10, help="IDK what this does")

    # TODO IMPLEMENT NUM_INTERACTS

    parser.add_argument("--save_location", type=str, default="results",
                        help="Which directory should results be saved in?")

    parser.add_argument("--network", choices=["DNN", "CNN"], default="DNN")
    args = parser.parse_args()

    return TrainingParameters(agent_name=args.agent_name,
                              env_name=args.env_name,
                              num_episodes=args.num_episodes,
                              save_location=args.save_location,
                              network=args.network)


# agent_name = sys.argv[1]
# game = sys.argv[2]
# interacts = int(sys.argv[3])
# def arg_spoof():
#     args = get_args()
#     arg_spoofs = []
#     for game_name in game_names:
#         for agent_name in agent_names:
#             print(game_name)
#             args_copy = copy.deepcopy(args)
#             args_copy.game_name = game_name
#             args_copy.agent_name = agent_name
#             args_copy.interacts = 1
#             arg_spoofs.append(args_copy)
#     return arg_spoofs


def train_from_params(train_params : TrainingParameters,
                      session_pref=None,
                      show_ep_prog_bar=False):
    device = torch.device("cpu")

    env, env_params = get_env_and_params(train_params.env_name)

    agent, mode = make_agent(agent_name=train_params.agent_name,
                             in_size=env_params["in_size"],
                             action_size=env_params["action_size"],
                             hidden=env_params["hid"],
                             network=train_params.network,
                             continuous=env_params["cont"],
                             tab_q_init=env_params["tab_q_init"])

    os.makedirs('./{}/{}/{}'.format(train_params.save_location, train_params.env_name, train_params.agent_name), exist_ok=True)
    process_id = str(time.time())[-5:]
    seed = int(process_id)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    datetime_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_pref = './{}/{}/{}/{}-{}-{}'.format(train_params.save_location, train_params.env_name, train_params.agent_name,
                                             train_params.agent_name, datetime_string, process_id)

    if train_params.num_episodes == -1:
        train_params.num_episodes = env_params["estimated_ep_required"]

    return run_episodic(agent, env, train_params.num_episodes, train_params, env_params["max_ep_length"],
                        mode, train_params.save_location, device=device,
                        int_action=env_params["int_action"], file_pref=file_pref,
                        tb_log_path=session_pref, agent_name=train_params.agent_name,
                        env_name=train_params.env_name,
                        show_ep_prog_bar=show_ep_prog_bar)


if __name__ == "__main__":
    params = get_train_params_from_args()
    train_from_params(params, show_ep_prog_bar=True)
