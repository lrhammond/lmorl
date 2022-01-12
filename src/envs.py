import numpy as np
import gym_safety # May falsely display as unused
import gym
from src.constants import env_names


class Env:
    state_repr_size = "not implemented"
    action_size = "not implemented"
    rec_hid_width = "not implemented"
    is_action_cont = "not implemented"
    rec_ep_length = "not implemented"
    rec_tabular_q_init = "not implemented"
    rec_episodes = "not implemented"

    def __init__(self, seed=42):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class CartSafe(Env):
    state_repr_size = 4
    action_size = 2
    rec_hid_width = 32  # TODO - how do we change this between models? add to params?
    is_action_cont = False
    rec_ep_length = 300
    rec_tabular_q_init = 300
    rec_episodes = 40000
    rec_interacts = int(1e5)

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


class GridNav(Env):
    state_repr_size = 625
    action_size = 4
    rec_hid_width = 128
    is_action_cont = False
    rec_ep_length = 50
    rec_tabular_q_init = 300
    rec_episodes = 20000
    rec_interacts = int(1e5)

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


class ThreeArmedBandit(Env):
    rec_hid_width = 8
    state_repr_size = 1
    action_size = 3
    is_action_cont = False
    rec_ep_length = 100
    rec_tabular_q_init = 100
    rec_episodes = 1000
    rec_interacts = int(1e4)

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
        reward = self.mus[int(action)] + np.random.uniform(low=-0.2, high=0.2)

        # next_state, reward, done, info
        return 0, reward, (self.episode_interacts >= 100), {}


env_dict = {
    'Bandit': ThreeArmedBandit,
    'CartSafe': CartSafe,
    'GridNav': GridNav
}
assert(list(env_dict.keys()) == env_names)

def get_env_by_name(game_name):
    return env_dict[game_name]()
