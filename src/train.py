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
import inspect
import torch
import random
import time
import os
import argparse
from tqdm import tqdm

import gym
import src.constants as constants

gym.logger.set_level(40)

from torch.utils.tensorboard import SummaryWriter

##################################################
from src.make_agent import make_agent, agent_names
from src.envs import *


##################################################

class TrainingParameters:
    def __init__(self,
                 agent_name: str,
                 env_name: str,
                 num_episodes: int,
                 network: str = "DNN",
                 test_group_label: str = None
                 ):
        print(agent_name)
        assert (agent_name in agent_names)
        assert (env_name in env_names)
        assert (network in ["CNN", "DNN"])
        self.agent_name = agent_name
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.network = network
        self.test_group_label = test_group_label

    def render_and_print(self):
        print(self.render_to_string())

    def render_to_string(self):
        x = ""
        for atr_name, atr in inspect.getmembers(self):
            if not atr_name.startswith("_") and not inspect.ismethod(atr):
                x += f" < {atr_name}: {str(atr)} >, "
        return x

    def render_to_file(self, dir):
        x = self.render_to_string()
        with open(dir, "w") as f: f.write(x)


def train_from_params(train_params: TrainingParameters,
                      session_pref: str,
                      show_ep_prog_bar=False):

    device = torch.device("cpu")

    env = get_env_by_name(train_params.env_name)

    agent, mode = make_agent(agent_name=train_params.agent_name,
                             env=env,
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

    # TODO - Add ep length and num episodes to argparse + Trainparameters

    if train_params.num_episodes == -1:
        train_params.num_episodes = env.rec_episodes

    max_ep_length = env.rec_ep_length

    if show_ep_prog_bar:
        episode_iter = tqdm(range(train_params.num_episodes))
    else:
        episode_iter = range(train_params.num_episodes)
    total_interacts = 0

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

            if env.is_action_cont:
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


        # if i % 5000 == 0:
        #    agent.save_model('./{}/{}/{}/{}-{}-{}'.format(save_location, game, agent_name, agent_name, process_id, i))

        writer.add_scalar(f"{train_params.env_name}/Reward", cum_reward, i)

        writer.add_scalar(f"{train_params.env_name}/Cost", cum_cost, i)

    agent.save_model(run_dir)
    writer.flush()


##################################################

def get_train_params_from_args():

    # agent_name, game, interacts = 'LPPO', 'CartSafe', 10000
    parser = argparse.ArgumentParser(description="Run lexico experiments")

    parser.add_argument("--agent_name", type=str, default="tabular", choices=agent_names,
                        help="The name of the type of agent (e.g AC, DQN, LDQN)")

    parser.add_argument("--env_name", type=str, default="Gaussian", choices=env_names,
                        help="The name of the game to train on e.g. 'MountainCarSafe', 'Gaussian', 'CartSafe':")

    parser.add_argument("--num_episodes", type=int, default=10, help="IDK what this does")

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
                      session_pref=session_pref,
                      show_ep_prog_bar=True)
