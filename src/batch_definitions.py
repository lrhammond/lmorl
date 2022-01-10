from src.make_agent import disc_agent_names
from src.train import TrainingParameters
from src.envs import env_names

batch_definitions = {

    "exit_tests": [
        TrainingParameters(
            agent_name=agent_name,
            env_name=env_name,
            num_episodes=1,
            test_group_label="exit_tests")

        for agent_name in disc_agent_names for env_name in env_names
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
