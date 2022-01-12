# Main file for running experiments

import os

# import safety_gym

# from open_safety.envs.puck_env import PuckEnv

##################################################
from make_agent import make_agent
from learners_other import *
from envs import *

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
from make_env import get_env_and_params

device = torch.device("cpu")

##################################################


def run(agent, game, mode, int_action=False):
    if game == 'BalanceBotEnv':
        if render:
            env = BalanceBot(render=True)
        else:
            env = BalanceBot(render=False)
    elif game == 'PuckEnv':
        env = PuckEnv()
    elif game == 'MountainCarSafe':
        env = MountainCarSafe()
    else:
        env = gym.make(game + '-v0')

    state = env.reset()
    state = np.expand_dims(state, axis=0)
    state = torch.tensor(state).float().to(device)

    if render:
        env.render()
    i = 0

    cumulative_cost = 0
    cumulative_reward = 0

    cumulative_costs = []
    cumulative_rewards = []

    for step in range(interacts):

        action = agent.act(state)
        i += 1

        # action = int(action)
        if type(action) != int:
            action = action.squeeze().cpu()
            if int_action:
                action = int(action)

        # print(action)
        ##        if step % 10 == 0:
        ##            print()
        ##            print(action)
        ##            for _ in range(5):
        ##                print(float(agent.act(state)))
        ##            print()

        next_state, reward, done, info = env.step(action)
        next_state = np.expand_dims(next_state, axis=0)
        next_state = torch.tensor(next_state).float().to(device)

        if render:
            env.render()

        try:
            cost = info['cost']
        except:
            try:
                cost = info['constraint_costs'][0]
            except:
                cost = 0

        cumulative_cost += cost
        cumulative_reward += reward

        if done or i >= max_step:

            # print(state)
            # print(next_state)
            # print()

            print(i)

            # print('cumulative reward: {}'.format(cumulative_reward))
            # print('cumulative cost: {}'.format(cumulative_cost))

            cumulative_costs.append(cumulative_cost)
            cumulative_rewards.append(cumulative_reward)

            if len(cumulative_costs) % 10 == 0:
                print(len(cumulative_costs))

            cumulative_cost = 0
            cumulative_reward = 0

            i = 0
            state = env.reset()
            state = np.expand_dims(state, axis=0)
            state = torch.tensor(state).float().to(device)
        else:
            state = next_state

    print()
    print('episodes: {}'.format(len(cumulative_rewards)))
    print('mean reward: {}'.format(sum(cumulative_rewards) / len(cumulative_rewards)))
    print('mean cost: {}'.format(sum(cumulative_costs) / len(cumulative_costs)))

    env.close()


##################################################
game_names = ["CartSafe",
              "GridNav",
              "MountainCarContinuousSafe",
              "PuckEnv",
              "BalanceBotEnv",
              "MountainCar",
              "MountainCarSafe"]
agent_names = []
agent_name = 'LPPO'
game = 'CartSafe'
version = 0

render = True

interacts = 100000
max_step = 300
int_action = False

env, env_params = get_env_and_params(game)
agent, mode = make_agent(agent_name,
                         env_params.in_size,
                         env_params.action_size,
                         env_params.hid, 'DNN',
                         env_params.cont, alt_lex=False)

i = 0
has_loaded = False
for _, _, files in os.walk('../results/{}/{}/'.format(game, agent_name)):
    for file in filter(lambda file: '.txt' in file, files):
        if i == version:
            print('../results/{}/{}/{}'.format(game, agent_name, file[:-4]))
            agent.load_model('../results/{}/{}/{}-3000000'.format(game, agent_name, file[:-4]))
            has_loaded = True
            break
        i += 1
if not has_loaded:
    print('using random initialisation!')

# t = '1616587665.7254992'
# agent.actor.load_state_dict(torch.load('/Users/lewishammond/Repositories/code/lmorl/actor_nan_error_{}.pt'.format(t)))
# agent.critic.load_state_dict(torch.load('/Users/lewishammond/Repositories/code/lmorl/critic_nan_error_{}.pt'.format(t)))
# with open('nan_error_data_{}.pickle'.format(t), "rb") as input_file:
#     p = pickle.load(input_file) 

##
##x = []
##y = []
##colours = []
##
##for pos in np.arange(-1.2, 0.6, 0.03):
##    for vel in np.arange(-0.07, 0.07, 0.003):
##        
##        x.append(pos)
##        y.append(vel)
##
##        s = np.asarray([pos, vel])
##        s = np.expand_dims(s, axis=0)
##        s = torch.tensor(s).float().to(device)
##        
##        lst = [agent.act(s) for _ in range(10)]
##        action = max(set(lst), key=lst.count)
##        if action == 0:
##            colours.append('r')
##        if action == 1:
##            colours.append('y')
##        if action == 2:
##            colours.append('g')
##
##plt.scatter(x,y,c=colours)
##plt.show()

##for param in agent.critic.parameters():
##    print(param.shape)
##    #print(param.data)
##    print(True in torch.isnan(param.data))
##
##print()
##print('#################')
##print()
##
##for param in agent.actor.parameters():
##    print(param.shape)
##    #print(param.data)
##    print(True in torch.isnan(param.data))
##
##print()
##print('#################')
##print()

run(agent, game, mode, int_action)

# AC 64436
# rand 43575
# RCPO 28656
