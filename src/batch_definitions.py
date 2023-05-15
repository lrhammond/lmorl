from src.constants import disc_agent_names, lex_agents
from src.TrainingParameters import TrainingParameters
from src.constants import env_names

batch_definitions = {

    "exit_tests":
        [
            TrainingParameters(
                agent_name=agent_name,
                env_name=env_name,
                num_interacts=2,
                test_group_label="exit_tests")

            for agent_name in disc_agent_names for env_name in env_names
        ],

    "agent_tests": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=100,
            test_group_label="agent_tests")

        for agent_name in disc_agent_names for env_name in ["Bandit"]
    ],

    "learn_tests": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=-1,
            test_group_label="learn_tests")

        for agent_name in disc_agent_names for env_name in env_names
    ],

    "learn_tests_bandit_short": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=int(1e4),
            test_group_label="learn_tests")

        for agent_name in disc_agent_names for env_name in ["Bandit"]
    ],

    "learn_tests_CartSafe_short": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=int(1e4),
            test_group_label="learn_tests")

        for agent_name in disc_agent_names for env_name in ["CartSafe"]
    ],

    "learn_tests_bandit_short_reward_size_1": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=int(1e4),
            test_group_label="learn_tests",
            reward_size=1) # TODO change mode thingy

        for agent_name in disc_agent_names for env_name in ["Bandit"]
    ],

    "learn_tests_bandit_long": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=int(1e5),
            test_group_label="learn_tests")

        for agent_name in disc_agent_names for env_name in ["Bandit"]
    ],

    "humble_test": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=1,
            test_group_label="humble_test")

        for agent_name in ["tabular"] for env_name in ["Bandit"]
    ],

    "recreating_fig2_CartSafe_short": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=int(2e5),
            test_group_label="recreating_fig2_CartSafe_short")

        for agent_name in disc_agent_names for env_name in ["CartSafe"]
    ],

    "recreating_fig2_CartSafe_long": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=int(1e6),
            test_group_label="recreating_fig2_CartSafe_long")

        for agent_name in disc_agent_names for env_name in ["CartSafe"]
    ],

    "recreating_fig2_CartSafe_really_short": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=int(1e3),
            test_group_label="recreating_fig2_CartSafe_long")

        for agent_name in disc_agent_names for env_name in ["CartSafe"]
    ],

    "fig3_short": [
        TrainingParameters(
            agent_name="LDQN",
            env_name=env_name,
            num_interacts=int(1e5),
            test_group_label="fig3_short",
            slack=slack
        )

        for slack in [0.0, 0.2, 0.4, 0.6, 0.8, 1] for env_name in ["CartSafe"]
    ],

    "fig3_long": [
        TrainingParameters(
            agent_name="LDQN",
            env_name=env_name,
            num_interacts=int(1e6),
            test_group_label="fig3_long",
            slack=slack
        )

        for slack in [0.0, 0.2, 0.4, 0.6, 0.8, 1] for env_name in ["CartSafe"]
    ],

    "fig_2_extra_lexico": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_interacts=int(1e6),
            test_group_label="fig_2_extra_lexico")

        for agent_name in lex_agents for env_name in ["CartSafe"]
    ]

}
