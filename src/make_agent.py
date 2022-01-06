from src.learners_lexicographic import LexDQN, LexTabular, LexActorCritic
from src.learners_other import ActorCritic, DQN, RCPO, VaR_PG, VaR_AC, AproPO, Tabular, RandomAgent

agent_names = ["tabular", "DQN", "AC", "random", "LDQN", "LA2C", "LPPO", "AproPO", "RCPO", "VaR_AC"]

# Although more agents have been implemented, we disable them as they havent been tested
# agent_names = ['AC', 'DQN', 'LDQN','RCPO','VaR_PG','VaR_AC','AproPO', 'tabular','LexTabular','invLexTabular', 'random',
#                'LA2C', 'seqLA2C', 'LPPO', 'seqLPPO', 'LA2C2nd', 'seqLA2C2nd', 'LPPO2nd', 'seqLPPO2nd']

def make_agent(agent_name, in_size=60, action_size=4, hidden=256, network='DNN', continuous=False, alt_lex=False):
    prioritise_performance_over_safety = False

    assert (agent_name in agent_names)

    if agent_name == 'AC':
        agent = ActorCritic(action_size=action_size, in_size=in_size,
                            network=network, hidden=hidden, continuous=continuous)
        mode = 1

    elif agent_name == 'DQN':
        agent = DQN(action_size=action_size, in_size=in_size,
                    network=network, hidden=hidden)
        mode = 1

    elif agent_name == 'LDQN':
        agent = LexDQN(action_size=action_size, in_size=in_size, reward_size=2,
                       network=network, hidden=hidden)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'RCPO':
        agent = RCPO(action_size=action_size, constraint=0.1, in_size=in_size,
                     network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name == 'VaR_PG':
        agent = VaR_PG(action_size=action_size, alpha=1, beta=0.95, in_size=in_size,
                       network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name == 'VaR_AC':
        agent = VaR_AC(action_size=action_size, alpha=1, beta=0.95, in_size=in_size,
                       network=network, hidden=hidden, continuous=continuous)
        mode = 4

    elif agent_name == 'AproPO':
        constraints = [(0.3, 0.5), (0.0, 0.1)]
        agent = AproPO(action_size=action_size,
                       in_size=in_size,
                       constraints=constraints,
                       reward_size=2,
                       network=network,
                       hidden=hidden,
                       continuous=continuous)
        mode = 4

    elif agent_name == 'tabular':
        agent = Tabular(action_size=action_size)
        mode = 1

    elif agent_name == 'LexTabular':
        agent = LexTabular(action_size=action_size)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'invLexTabular':
        agent = LexTabular(action_size=action_size)
        mode = 5

    elif agent_name == 'random':
        agent = RandomAgent(action_size=action_size, continuous=continuous)
        mode = 1

    elif agent_name == 'LA2C':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'seqLA2C':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'LPPO':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'seqLPPO':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=False,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'LA2C2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'seqLA2C2nd':
        agent = ActorCritic(in_size=in_size, action_size=action_size, mode='a2c', second_order=True,
                            reward_size=2, network='DNN', hidden=hidden, sequential=True, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'LPPO2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=False, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'seqLPPO2nd':
        agent = LexActorCritic(in_size=in_size, action_size=action_size, mode='ppo', second_order=True,
                               reward_size=2, network='DNN', hidden=hidden, sequential=True, continuous=continuous)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    else:
        print('invalid agent specification')
        assert False

    return agent, mode