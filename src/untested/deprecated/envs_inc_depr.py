import gym, math, torch

import numpy as np
import pybullet as p

import gym_safety
from gym.envs.classic_control.mountain_car import MountainCarEnv
from open_safety.envs.balance_bot_env import BalanceBotEnv

import gym


class MountainCarSafe:

    def __init__(self):
        self.env = MountainCarEnv()  # gym.make('MountainCar-v0')

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        position, velocity = next_state

        info['cost'] = 0.0 if position > -1.15 else 1.0
        reward = 100.0 if (position >= 0.5 and velocity >= 0) else 0.0
        # if velocity > 0:
        #    reward += velocity**2

        return next_state, reward, done, info

    def close(self):
        self.env.close()


class BalanceBot:

    def __init__(self, objective="Distance", cost="Fall", render=False):
        self.env = BalanceBotEnv(objective, cost, render)

    def reset(self):
        self.previous_displacement = 0
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        if torch.numel(action) == 1:
            action = torch.tensor([action, action])
        next_state, reward, done, info = self.env.step(action)

        # cube_position, cube_orientation = p.getBasePositionAndOrientation(self.env.bot_id)
        # displacement = cube_position[1]
        # reward += max(0, displacement-self.previous_displacement)
        # self.previous_displacement = displacement

        return next_state, 10 * reward, done, info

    def close(self):
        self.env.close()

    # 'CartSafe': {
    #     "env": CartSafe,
    #     "in_size": 4,
    #     "action_size": 2,
    #     "hid": 8,
    #     "int_action": True,
    #     "cont": False,
    #     "max_ep_length": 300,
    #     "tab_q_init": 300,
    #     "estimated_ep_required": 40000},


class CartSafe:
    state_repr_size = 4
    action_size = 2
    rec_hid_width = 8
    is_action_cont = False
    rec_ep_length = 300
    rec_tabular_q_init = 300
    rec_episodes = 40000

    def __init__(self):
        self.env = gym.make('CartSafe-v0')

    def reset(self):
        self.i = 0
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        self.i += 1
        next_state, reward, done, info = self.env.step(int(action))
        cart_pos = next_state[0]

        if abs(cart_pos) >= 2.39:
            info['constraint_costs'][0] += max(300 - self.i, 0)
            done = True

        return next_state, reward, done, info

    def close(self):
        self.env.close()


class GridNav:
    state_repr_size = 625
    action_size = 4
    rec_hid_width = 128
    is_action_cont = False
    rec_ep_length = 50
    rec_tabular_q_init = 300
    rec_episodes = 20000

    def __init__(self, seed=42):
        self.env = gym.make('GridNav-v0')
        # self.env.customize_game(seed=seed, stochasticity=0.0)

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
        next_state, reward, done, info = self.env.step(int(action))

        reward = 100 if reward == 1000 else 0

        return next_state, reward, done, info

    def close(self):
        self.env.close()


class Simple1DEnv:

    def __init__(self):
        self.mu = np.random.uniform(low=-2.0, high=2.0)
        self.var = np.random.uniform(low=0.1, high=1.0)

    def reset(self):
        return 0

    def render(self):
        print('#############################')
        print(self.mu)
        print(self.var)
        print('#############################')

    # TODO Gaussian is broken?

    def step(self, action):
        reward = math.e ** (-action ** 2 / 2) / (2 * math.pi) ** 0.5

        return 0, reward, False, {}


class ThreeArmedBandit:
    rec_hid_width = 8
    state_repr_size = 1
    action_size = 3
    is_action_cont = False
    rec_ep_length = 100
    rec_tabular_q_init = 100
    rec_episodes = 1000

    def __init__(self):
        # We randomly generate the means of the payoffs
        # To enable learning between episodes,
        #  setting the last arm to have better payoff in expectation
        self.mus = [0.3, 0.5, 0.7]
        self.episode_interacts = 0

    def reset(self):
        self.episode_interacts = 0
        return 0

    def render(self):
        print('#############################')
        print(self.mus)
        print('#############################')

    def step(self, action):
        self.episode_interacts += 1
        assert (action in [0, 1, 2])
        reward = self.mus[int(action)] + np.random.uniform(low=-0.3, high=0.3)

        # next_state, reward, done, info
        return 0, reward, (self.episode_interacts >= 100), {}


class SimpleContinuousAction:

    def __init__(self):
        self.episode_interacts = 0

    def reset(self):
        self.episode_interacts = 0
        return 0

    def render(self):
        print('#############################')
        print("A simple cont action")
        print('#############################')

    def step(self, action):
        reward = math.e ** (- (action - 0.5) ** 2)
        self.episode_interacts += 1

        # next_state, reward, done, info
        return 0, reward, (self.episode_interacts >= 100), {}


# TODO - probably move these to be attributes of the env objects?
# TODO - replace "tab_q_init" with max reward, use to calculate tab_q_init equivalent
env_dict = {
    'Bandit': ThreeArmedBandit,
    'CartSafe': CartSafe,
    'GridNav': GridNav
    # {
    # "env": ThreeArmedBandit,
    # "hid": 8,
    # "action_size": 3,
    # "in_size": 1,
    # "int_action": True,
    # "cont": False,
    # "max_ep_length": 100,
    # "tab_q_init": 100,
    # "estimated_ep_required": 1000},

    # 'SimpleContinuousAction': {
    #     "env": SimpleContinuousAction,
    #     "hid": 4,
    #     "action_size": 1,
    #     "in_size": 1,
    #     "int_action": False,
    #     "cont": True,
    #     "max_ep_length": 100,
    #     "tab_q_init": None,
    #     "estimated_ep_required": 1000},

    # 'Gaussian': {
    #     "env": Simple1DEnv,
    #     "hid": 8,
    #     "action_size": 1,
    #     "in_size": 1,
    #     "int_action": False,
    #     "cont": True,
    #     "max_ep_length": 100},

    # 'CartSafe': {
    #     "env": CartSafe,
    #     "in_size": 4,
    #     "action_size": 2,
    #     "hid": 8,
    #     "int_action": True,
    #     "cont": False,
    #     "max_ep_length": 300,
    #     "tab_q_init": 300,
    #     "estimated_ep_required": 40000},
    #
    # 'GridNav': {
    #     "env": (lambda : gym.make('GridNav-v0')),
    #     "in_size": 625,
    #     "action_size": 4,
    #     "hid": 128,
    #     "int_action": True,
    #     "cont": False,
    #     "max_ep_length": 50,
    #     "tab_q_init": 300,
    #     "estimated_ep_required": 20000}

    # The remaining envs are untested and so disabled
    # 'MountainCarContinuousSafe': {
    #     "env": None,
    #     "in_size": 2,
    #     "action_size": 2,
    #     "hid": 16,
    #     "cont": True,
    #     "int_action": False,
    #     "max_ep_length": 200},
    #
    # 'PuckEnv': {
    #     "env": None,
    #     "in_size": 18,
    #     "action_size": 2,
    #     "int_action": False,
    #     "hid": 128,
    #     "cont": True},
    #
    # 'BalanceBotEnv': {
    #     "env": BalanceBot,
    #     "in_size": 32,
    #     "action_size": 2,
    #     "int_action": False,
    #     "hid": 32,
    #     "cont": True,
    #     "max_ep_length": 300},
    #
    # 'MountainCar': {
    #     "env": None,
    #     "in_size": 2,
    #     "action_size": 3,
    #     "hid": 32,
    #     "cont": False,
    #     "int_action": True,
    #     "max_ep_length": 200},
    #
    # 'MountainCarSafe': {
    #     "env": MountainCarSafe,
    #     "in_size": 2,
    #     "action_size": 3,
    #     "hid": 32,
    #     "cont": False,
    #     "int_action": True,
    #     "max_ep_length": 300}
}

env_names = list(env_dict.keys())


# def get_env_from_gym(env_name):
#
#     return env

def get_env_by_name(game_name):
    return env_dict[game_name]()
