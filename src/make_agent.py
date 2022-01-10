from src.learners_lexicographic import LexDQN, LexTabular, LexActorCritic
from src.learners_other import ActorCritic, DQN, RCPO, VaR_PG, VaR_AC, AproPO, Tabular, RandomAgent
from src.envs import Env

agent_names = ["tabular", "DQN", "AC", "random", "LDQN", "LA2C", "LPPO", "AproPO", "RCPO", "VaR_AC"]

# TODO - add cont list, add checks for agent-env compat

disc_agent_names = [
    "tabular",
    "DQN",
    "AC",
    "random",
    "LDQN",
    "LA2C",
    "LPPO",
    "AproPO",
    "RCPO",
    "VaR_AC"
]


def make_agent(agent_name: str, env: Env, network: str = 'DNN'):
    prioritise_performance_over_safety = False

    assert (agent_name in agent_names)

    if agent_name == 'AC':
        agent = ActorCritic(action_size=env.action_size, in_size=env.state_repr_size,
                            network=network, hidden=env.rec_hid_width, is_action_cont=env.is_action_cont)
        mode = 1

    elif agent_name == 'DQN':
        agent = DQN(action_size=env.action_size, in_size=env.state_repr_size,
                    network=network, hidden=env.rec_hid_width)
        mode = 1

    elif agent_name == 'LDQN':
        agent = LexDQN(action_size=env.action_size, in_size=env.state_repr_size, reward_size=2,
                       network=network, hidden=env.rec_hid_width)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'RCPO':
        agent = RCPO(action_size=env.action_size, constraint=0.1, in_size=env.state_repr_size,
                     network=network, hidden=env.rec_hid_width, is_action_cont=env.is_action_cont)
        mode = 4

    elif agent_name == 'VaR_PG':
        agent = VaR_PG(action_size=env.action_size, alpha=1, beta=0.95, in_size=env.state_repr_size,
                       network=network, hidden=env.rec_hid_width, is_action_cont=env.is_action_cont)
        mode = 4

    elif agent_name == 'VaR_AC':
        agent = VaR_AC(action_size=env.action_size, alpha=1, beta=0.95, in_size=env.state_repr_size,
                       network=network, hidden=env.rec_hid_width, is_action_cont=env.is_action_cont)
        mode = 4

    elif agent_name == 'AproPO':
        constraints = [(0.3, 0.5), (0.0, 0.1)]
        agent = AproPO(action_size=env.action_size,
                       in_size=env.state_repr_size,
                       constraints=constraints,
                       reward_size=2,
                       network=network,
                       hidden=env.rec_hid_width,
                       is_action_cont=env.is_action_cont)
        mode = 4

    elif agent_name == 'tabular':
        agent = Tabular(action_size=env.action_size, initialisation=env.rec_tabular_q_init)
        mode = 1

    elif agent_name == 'LexTabular':
        agent = LexTabular(action_size=env.action_size)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'invLexTabular':
        agent = LexTabular(action_size=env.action_size)
        mode = 5

    elif agent_name == 'random':
        agent = RandomAgent(action_size=env.action_size, is_action_cont=env.is_action_cont)
        mode = 1

    elif agent_name == 'LA2C':
        agent = LexActorCritic(in_size=env.state_repr_size, action_size=env.action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=env.rec_hid_width, sequential=False,
                               is_action_cont=env.is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'seqLA2C':
        agent = LexActorCritic(in_size=env.state_repr_size, action_size=env.action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=env.rec_hid_width, sequential=True,
                               is_action_cont=env.is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'LPPO':
        agent = LexActorCritic(in_size=env.state_repr_size, action_size=env.action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=env.rec_hid_width, sequential=False,
                               is_action_cont=env.is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'seqLPPO':
        agent = LexActorCritic(in_size=env.state_repr_size, action_size=env.action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=env.rec_hid_width, sequential=True,
                               is_action_cont=env.is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'LA2C2nd':
        agent = LexActorCritic(in_size=env.state_repr_size, action_size=env.action_size, mode='a2c', second_order=True,
                               reward_size=2, network='DNN', hidden=env.rec_hid_width, sequential=False,
                               is_action_cont=env.is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'seqLA2C2nd':
        agent = ActorCritic(in_size=env.state_repr_size, action_size=env.action_size, mode='a2c', second_order=True,
                            reward_size=2, network='DNN', hidden=env.rec_hid_width, sequential=True,
                            is_action_cont=env.is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'LPPO2nd':
        agent = LexActorCritic(in_size=env.state_repr_size, action_size=env.action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=env.rec_hid_width, sequential=False,
                               is_action_cont=env.is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'seqLPPO2nd':
        agent = LexActorCritic(in_size=env.state_repr_size, action_size=env.action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=env.rec_hid_width, sequential=True,
                               is_action_cont=env.is_action_cont)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    else:
        print('invalid agent specification')
        assert False

    return agent, mode
