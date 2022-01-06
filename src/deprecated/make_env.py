import gym

from src.envs import CartSafe, BalanceBot, MountainCarSafe, Simple1DEnv

env_dict = {
    'Gaussian': {
        "env": Simple1DEnv,
        "hid": 8,
        "action_size": 1,
        "int_action": False,
        "in_size": 1,
        "cont": True,
        "max_ep_length": 200},

    'CartSafe': {
        "env": CartSafe,
        "in_size": 4,
        "action_size": 2,
        "hid": 8,
        "cont": False,
        "int_action": True,
        "max_ep_length": 300},

    'GridNav': {
        "env": None,
        "in_size": 625,
        "action_size": 4,
        "hid": 128,
        "cont": False,
        "int_action": True,
        "max_ep_length": 50},

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

def get_env_and_params(game_name):

    env_params = env_dict[game_name]

    if env_params["env"] is not None:
        env = env_params["env"]()
    else:
        env = gym.make(game_name + '-v0')
        try:
            env_params["action_size"] = env.action_space.n
        except:
            env_params["action_size"] = len(env.action_space.high)
        env_params["in_size"] = len(env.observation_space.high)

    return env, env_params