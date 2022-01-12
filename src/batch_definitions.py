from src.constants import disc_agent_names
from src.TrainingParameters import TrainingParameters
from src.constants import env_names

batch_definitions = {

    "exit_tests":
        [
            TrainingParameters(
                agent_name=agent_name,
                env_name=env_name,
                num_episodes=1,
                test_group_label="exit_tests")

            for agent_name in disc_agent_names for env_name in env_names
        ]
        +
        [ # Test num_interacts as well
            TrainingParameters(
                agent_name=agent_name,
                env_name=env_name,
                num_interacts=10,
                test_group_label="exit_tests")

            for agent_name in disc_agent_names[0:2] for env_name in env_names[0:2]
        ],

    "interact_tests":
        [
            TrainingParameters(
                agent_name=disc_agent_names[0],
                env_name=env_names[0],
                num_interacts=num_interacts,
                test_group_label="exit_tests")

            for num_interacts in [314, 99, 100, 0]
        ],

    "agent_tests": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_episodes=10,
            test_group_label="agent_tests")

        for agent_name in disc_agent_names for env_name in ["Bandit"]
    ],

    "learn_tests": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_episodes=-1,
            test_group_label="learn_tests")

        for agent_name in disc_agent_names for env_name in env_names
    ],

    "humble_test": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_episodes=1,
            test_group_label="humble_test")

        for agent_name in ["tabular"] for env_name in ["Bandit"]
    ],

}
