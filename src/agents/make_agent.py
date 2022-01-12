from src.TrainingParameters import TrainingParameters
from src.constants import disc_agent_names
from src.agents.learners_other import ActorCritic, DQN, RCPO, VaR_AC, AproPO, Tabular, RandomAgent  # , VaR_PG
from src.agents.learners_lexicographic import LexDQN, LexActorCritic
from src.envs import Env


def make_agent(agent_name: str,
               env: Env,
               train_params: TrainingParameters):
    prioritise_performance_over_safety = False

    assert (agent_name in disc_agent_names)

    if agent_name == 'AC':
        agent = ActorCritic(train_params=train_params,
                            action_size=env.action_size,
                            in_size=env.state_repr_size,
                            hidden=env.rec_hid_width,
                            is_action_cont=env.is_action_cont)
        mode = 1

    elif agent_name == 'DQN':
        agent = DQN(train_params=train_params,
                    action_size=env.action_size,
                    in_size=env.state_repr_size,
                    hidden=env.rec_hid_width)
        mode = 1

    elif agent_name == 'LDQN':
        agent = LexDQN(train_params=train_params,
                       action_size=env.action_size,
                       in_size=env.state_repr_size,
                       hidden=env.rec_hid_width)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'RCPO':
        agent = RCPO(train_params=train_params,
                     action_size=env.action_size,
                     in_size=env.state_repr_size,
                     hidden=env.rec_hid_width,
                     is_action_cont=env.is_action_cont)
        mode = 4

    elif agent_name == 'VaR_AC':
        agent = VaR_AC(train_params=train_params,
                       action_size=env.action_size,
                       in_size=env.state_repr_size,
                       hidden=env.rec_hid_width,
                       is_action_cont=env.is_action_cont)
        mode = 4

    elif agent_name == 'AproPO':
        agent = AproPO(train_params=train_params,
                       in_size=env.state_repr_size,
                       action_size=env.action_size,
                       hidden=env.rec_hid_width,  # TODO - move this to TrainParam
                       is_action_cont=env.is_action_cont)
        mode = 4

    elif agent_name == 'tabular':
        agent = Tabular(train_params, action_size=env.action_size,
                        initialisation=env.rec_tabular_q_init)
        mode = 1

    elif agent_name == 'random':
        agent = RandomAgent(train_params, action_size=env.action_size,
                            is_action_cont=env.is_action_cont)
        mode = 1

    elif agent_name == 'LA2C':
        agent = LexActorCritic(train_params,
                               in_size=env.state_repr_size,
                               action_size=env.action_size,
                               mode='a2c',
                               hidden=env.rec_hid_width,
                               is_action_cont=env.is_action_cont,
                               second_order=False,
                               sequential=False,
                               extra_input=False)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    elif agent_name == 'LPPO':

        agent = LexActorCritic(train_params=train_params,
                               in_size=env.state_repr_size,
                               action_size=env.action_size,
                               mode='ppo',
                               second_order=False,
                               sequential=False,
                               is_action_cont=env.is_action_cont,
                               extra_input=False,
                               hidden=env.rec_hid_width)
        if prioritise_performance_over_safety:
            mode = 5
        else:
            mode = 3

    else:
        print('invalid agent specification')
        assert False

    return agent, mode

    # Untested
    #
    # elif agent_name == 'LexTabular':
    #     agent = LexTabular(action_size=env.action_size)
    #     if prioritise_performance_over_safety:
    #         mode = 5
    #     else:
    #         mode = 3
    #
    # elif agent_name == 'invLexTabular':
    #     agent = LexTabular(action_size=env.action_size)
    #     mode = 5
    #
    # elif agent_name == 'seqLA2C':
    #     agent = LexActorCritic(in_size=env.state_repr_size,
    #                            action_size=env.action_size,
    #                            mode='a2c',
    #                            second_order=False,
    #                            reward_size=2,
    #                            network='DNN',
    #                            hidden=env.rec_hid_width,
    #                            sequential=True,
    #                            is_action_cont=env.is_action_cont)
    #     if prioritise_performance_over_safety:
    #         mode = 5
    #     else:
    #         mode = 3
    #
    # elif agent_name == 'seqLPPO':
    #     agent = LexActorCritic(in_size=env.state_repr_size,
    #                            action_size=env.action_size,
    #                            mode='ppo',
    #                            second_order=False,
    #                            reward_size=2,
    #                            network='DNN',
    #                            hidden=env.rec_hid_width,
    #                            sequential=True,
    #                            is_action_cont=env.is_action_cont)
    #     if prioritise_performance_over_safety:
    #         mode = 5
    #     else:
    #         mode = 3
    #
    # elif agent_name == 'LA2C2nd':
    #     agent = LexActorCritic(in_size=env.state_repr_size,
    #                            action_size=env.action_size,
    #                            mode='a2c',
    #                            second_order=True,
    #                            reward_size=2,
    #                            network='DNN',
    #                            hidden=env.rec_hid_width,
    #                            sequential=False,
    #                            is_action_cont=env.is_action_cont)
    #     if prioritise_performance_over_safety:
    #         mode = 5
    #     else:
    #         mode = 3
    #
    # elif agent_name == 'seqLA2C2nd':
    #     agent = ActorCritic(in_size=env.state_repr_size,
    #                         action_size=env.action_size,
    #                         mode='a2c',
    #                         second_order=True,
    #                         reward_size=2,
    #                         network='DNN',
    #                         hidden=env.rec_hid_width,
    #                         sequential=True,
    #                         is_action_cont=env.is_action_cont)
    #     if prioritise_performance_over_safety:
    #         mode = 5
    #     else:
    #         mode = 3
    #
    # elif agent_name == 'LPPO2nd':
    #     agent = LexActorCritic(in_size=env.state_repr_size,
    #                            action_size=env.action_size,
    #                            mode='ppo',
    #                            second_order=True,
    #                            reward_size=2,
    #                            network='DNN',
    #                            hidden=env.rec_hid_width,
    #                            sequential=False,
    #                            is_action_cont=env.is_action_cont)
    #     if prioritise_performance_over_safety:
    #         mode = 5
    #     else:
    #         mode = 3
    #
    # elif agent_name == 'seqLPPO2nd':
    #     agent = LexActorCritic(in_size=env.state_repr_size,
    #                            action_size=env.action_size,
    #                            mode='ppo',
    #                            second_order=True,
    #                            reward_size=2,
    #                            network='DNN',
    #                            hidden=env.rec_hid_width,
    #                            sequential=True,
    #                            is_action_cont=env.is_action_cont)
    #
    #     if prioritise_performance_over_safety:
    #         mode = 5
    #     else:
    #         mode = 3

    # elif agent_name == 'VaR_PG':
    #     agent = VaR_PG(action_size=env.action_size,
    #                    alpha=train_params.alpha1,
    #                    beta=0.95,
    #                    in_size=env.state_repr_size,
    #                    network=network,
    #                    hidden=env.rec_hid_width,
    #                    is_action_cont=env.is_action_cont)
    #     mode = 4