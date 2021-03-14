import gym, math
import numpy as np
from gym.envs.classic_control.mountain_car import MountainCarEnv

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
        reward = 10.0 if (position >= 0.5 and velocity >= 0) else 0.0
        reward += velocity

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
