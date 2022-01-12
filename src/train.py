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
from src.make_agent import make_agent
from src.constants import agent_names
from src.envs import *


class InteractIter:
    # Instantiates the iterator class
    # Is stateful, but simplifies the control flow in training

    def __init__(self, max_interactions: int):
        self._max_interactions = max_interactions
        self._cur_interactions = 0
        self._cur_iterations = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._max_interactions >= self._cur_interactions:
            self._cur_iterations += 1
            return self._cur_iterations
        else:
            raise StopIteration()

    def increment(self):
        self._cur_interactions += 1


def train_from_params(train_params: TrainingParameters,
                      session_pref: str,
                      show_ep_prog_bar=True):
    device = torch.device("cpu")

    env = get_env_by_name(train_params.env_name)

    agent, mode = make_agent(agent_name=train_params.agent_name,
                             env=env,
                             train_params=train_params,
                             network=train_params.network)

    process_id = str(time.time())[-5:]
    seed = int(process_id)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    run_dir = os.path.join(session_pref, train_params.env_name, train_params.agent_name + "-" + str(process_id))
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    train_params.render_to_file(run_dir + ".params")

    if train_params.num_episodes == -1:
        train_params.num_episodes = env.rec_episodes

    max_ep_length = env.rec_ep_length

    total_interacts = 0

    if not train_params.is_interact_mode:
        interact_iter = "Undefined"
        if show_ep_prog_bar:
            episode_iter = tqdm(range(train_params.num_episodes), colour="green")
        else:
            episode_iter = range(train_params.num_episodes)
    else:
        interact_iter = InteractIter(train_params.num_interacts)

        if show_ep_prog_bar:
            episode_iter = tqdm(interact_iter)
        else:
            episode_iter = interact_iter

    # ========== TRAINING LOOP ==========
    for i in episode_iter:
        state = env.reset()
        state = np.expand_dims(state, axis=0)
        state = torch.tensor(state).float().to(device)

        cum_reward = 0
        cum_cost = 0

        for j in range(max_ep_length):
            action = agent.act(state)

            total_interacts += 1
            if train_params.is_interact_mode:
                interact_iter.increment()

            if env.is_action_cont:
                action = int(action)
            if type(action) != int:
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

        if train_params.save_every_n is not None and i % train_params.save_every_n == 0:
            agent.save_model(run_dir)

        writer.add_scalar(f"{train_params.env_name}/Reward", cum_reward, i)

        writer.add_scalar(f"{train_params.env_name}/Cost", cum_cost, i)

    agent.save_model(run_dir)
    writer.flush()


##################################################

def get_train_params_from_args():
    # TODO - add option for num interacts
    parser = argparse.ArgumentParser(description="Run lexico experiments")

    parser.add_argument("--agent_name", type=str, default="tabular", choices=agent_names,
                        help="The name of the type of agent")

    parser.add_argument("--env_name", type=str, default="Bandit", choices=env_names,
                        help="The name of the game to train on")

    parser.add_argument("--num_episodes", type=int, default=10, help="The number of episodes to train on")

    # TODO IMPLEMENT NUM_INTERACTS

    parser.add_argument("--network", choices=["DNN", "CNN"], default="DNN")
    args = parser.parse_args()

    return TrainingParameters(agent_name=args.agent_name,
                              env_name=args.env_name,
                              num_episodes=args.num_episodes,
                              network=args.network)


if __name__ == "__main__":
    params = get_train_params_from_args()
    session_pref = os.path.join(constants.data_dir, "misc",
                                datetime.now().strftime('%Y%m%d-%H%M%S'))

    train_from_params(params,
                      session_pref=session_pref)
