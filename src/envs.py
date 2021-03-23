import gym, math, torch

import numpy as np
import pybullet as p

import gym_safety
from gym.envs.classic_control.mountain_car import MountainCarEnv
from open_safety.envs.balance_bot_env import BalanceBotEnv


class MountainCarSafe:

    def __init__(self):
        self.env = MountainCarEnv() #gym.make('MountainCar-v0')

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):

        next_state, reward, done, info = self.env.step(action)
        position, velocity = next_state

        info['cost'] = 0.0 if position > -1.15 else 1.0
        reward = 100.0 if (position >= 0.5 and velocity >= 0) else 0.0
        #if velocity > 0:
        #    reward += velocity**2

        return next_state, reward, done, info

    def close(self):
        self.env.close()


class BalanceBot:

    def __init__(self,objective="Distance", cost="Fall", render=False):
        self.env = BalanceBotEnv(objective,cost,render)

    def reset(self):
        self.previous_displacement = 0
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):

        if torch.numel(action)==1:
            action = torch.tensor([action, action])
        next_state, reward, done, info = self.env.step(action)
        
        #cube_position, cube_orientation = p.getBasePositionAndOrientation(self.env.bot_id)
        #displacement = cube_position[1]
        #reward += max(0, displacement-self.previous_displacement)
        #self.previous_displacement = displacement

        return next_state, 10*reward, done, info

    def close(self):
        self.env.close()


class CartSafe:

    def __init__(self):
        self.env = gym.make('CartSafe-v0')

    def reset(self):
        self.i = 0
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):

        self.i += 1
        next_state, reward, done, info = self.env.step(action)
        cart_pos = next_state[0]

        if abs(cart_pos) >= 2.39:
            info['constraint_costs'][0] += max(300-self.i, 0)
            done = True

        return next_state, reward, done, info

    def close(self):
        self.env.close()


class GridNav:

    def __init__(self, seed=42):
        self.env = gym.make('GridNav-v0')
        #self.env.customize_game(seed=seed, stochasticity=0.0)

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):

        next_state, reward, done, info = self.env.step(action)

        reward = 100 if reward == 1000 else 0

        return next_state, reward, done, info

    def close(self):
        self.env.close()
        

class Simple1DEnv:

    def __init__(self):
        self.mu  = np.random.uniform(low=-1.0, high=1.0)
        self.var = np.random.uniform(low=0.1, high=1.0)

    def reset(self):
        return 0

    def render(self):
        print('#############################')
        print(self.mu)
        print(selv.var)
        print('#############################')

    def step(self, action):

        reward = math.e**(-action**2/2)/(2*math.pi)**0.5

        return 0, reward, False, {}
