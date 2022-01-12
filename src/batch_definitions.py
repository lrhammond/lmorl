from src.constants import disc_agent_names
from src.TrainingParameters import TrainingParameters
from src.constants import env_names

batch_definitions = {

    "exit_tests":
        [
            TrainingParameters(
                agent_name=agent_name,
                env_name=env_name,
                num_interacts=-1,
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

    # BUFFER_SIZE = int(1e5)
    # BATCH_SIZE = 8
    # UPDATE_EVERY = 8
    # UPDATE_EVERY_EPS = 1
    #
    # EPSILON = 0.05
    # SLACK = 0.04
    # LAMBDA_LR_2 = 0.05
    #
    # LR = 1e-3
    #
    # update_steps = 10
    #
    # network size: 4 x 32 x 32 x 2
    # max_ep_length = 300

    "CartSafe_LDQN": [
        TrainingParameters(
            agent_name="LDQN",
            env_name="CartSafe",
            num_interacts=1000000,
            test_group_label="CartSafe AC",
            buffer_size=int(1e5),
            batch_size=8,
            update_every=8,
            # update_every_eps=1 # Does nothing
            slack=0.04,
            reward_size=2
        )

        for i in range(0,10)
    ],

}
