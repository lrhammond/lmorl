# MAIN ORDER:
# TODO get training working on cluster
# TODO initial training - check everything learns
# TODO get VaR_AC learning
# TODO add hyperparameters (e.g. learning rate) to TrainingParameters
# TODO implement a discretisation for cartsafe etc with tabular
# TODO implement safety-gym
# TODO look for existing hyper param + models for safety-gym
# TODO get decent learning for safety environment (point+car, -- ,lvl1+2)

# Main file for running experiments
from datetime import datetime

import torch
import random
import time
import os
import argparse
from tqdm import tqdm

import gym
import src.constants as constants
from src.TrainingParameters import TrainingParameters

gym.logger.set_level(40)

from torch.utils.tensorboard import SummaryWriter
from src.agents.make_agent import make_agent
from src.constants import agent_names
from src.envs import *


class InteractIter:
    # Instantiates the iterator class
    # Is stateful, but simplifies the control flow in training

    def __init__(self, max_interactions: int):
        self._max_interactions = max_interactions
        self._cur_interactions = 0
        self._cur_episodes = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._max_interactions >= self._cur_interactions:
            self._cur_episodes += 1
            return self._cur_episodes
        else:
            raise StopIteration()

    def increment(self):
        self._cur_interactions += 1


def train_from_params(train_params: TrainingParameters,
                      session_pref: str,
                      show_prog_bar=True):
    device = torch.device("cpu")

    env = get_env_by_name(train_params.env_name)

    agent, mode = make_agent(agent_name=train_params.agent_name,
                             env=env,
                             train_params=train_params)

    process_id = str(time.time())[-5:]
    seed = int(process_id)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_dir = os.path.join(session_pref, train_params.env_name, train_params.agent_name + "-" + str(process_id))
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    train_params.render_to_file(run_dir + ".params")

    if train_params.num_episodes is not None:
        raise NotImplementedError("episodic mode is deprecated, use num_interacts instead")

    if train_params.num_interacts == -1:
        train_params.num_interacts = env.rec_interacts

    max_ep_length = env.rec_ep_length

    state = env.reset()
    state = np.expand_dims(state, axis=0)
    state = torch.tensor(state).float().to(device)

    interact_iter = range(train_params.num_interacts)
    if show_prog_bar:
        interact_iter = tqdm(interact_iter, colour="green", desc="Interacts")
    interacts_this_ep = 0
    for i in interact_iter:

        action = agent.act(state)
        interacts_this_ep += 1

        if not env.is_action_cont:
            action = int(action)
        if type(action) != int:
            action = action.squeeze().cpu().float()

        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = torch.tensor(next_state).float().to(device)

        if "cost" in info.keys():
            cost = info["cost"]
        elif "constraint_costs" in info.keys():
            cost = info['constraint_costs'][0]
        else:
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

        agent.step(state, action, r, next_state, done)

        if done or (interacts_this_ep >= max_ep_length):
            interacts_this_ep = 0
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state).float().to(device)
        else:
            state = next_state

        if train_params.save_every_n is not None and i % train_params.save_every_n == 0:
            agent.save_model(run_dir)

        writer.add_scalar(f"{train_params.env_name}/Reward", reward, i)
        writer.add_scalar(f"{train_params.env_name}/Cost", cost, i)

    agent.save_model(run_dir)
    writer.flush()


##################################################

def get_train_params_from_args():
    # TODO - add option for num interacts
    parser = argparse.ArgumentParser(description="Run lexico experiments")

    parser.add_argument("--env_name", type=str, default="Bandit", choices=env_names,
                        help="The name of the game to train on")

    parser.add_argument("--agent_name", type=str, default="tabular", choices=agent_names,
                        help="The name of the type of agent")

    parser.add_argument("--num_interacts", type=int, default=1e4, help="The number of interacts to train on")

    parser.add_argument("--network", choices=["DNN", "CNN"], default="DNN")
    args = parser.parse_args()

    return TrainingParameters(agent_name=args.agent_name,
                              env_name=args.env_name,
                              num_interacts=args.num_interacts,
                              network=args.network)


if __name__ == "__main__":
    params = get_train_params_from_args()
    session_pref = os.path.join(constants.data_dir, "misc",
                                datetime.now().strftime('%Y%m%d-%H%M%S'))

    train_from_params(params,
                      session_pref=session_pref)
