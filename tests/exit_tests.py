from src.train import TrainingParameters, train_from_params
from src.envs import env_names
from src.make_agent import agent_names, disc_agent_names
from tqdm import tqdm

pairs = [(an, en) for an in disc_agent_names for en in env_names]
successes = []
failures = []

print("Running sanity tests")
pbar = tqdm(pairs)
for agent_name, env_name in pbar:
    pbar.set_description(f"{len(successes)}S:{len(failures)}F")

    train_params = TrainingParameters(agent_name=agent_name, env_name=env_name,
                                      num_episodes=1, save_location="sanity_test_logs")
    try:
        train_from_params(train_params)
        successes.append((agent_name, env_name))
    except Exception as e:
        failures.append((agent_name, env_name, e))


print(f"Final total: {len(successes)}S:{len(failures)}F")

if len(failures) > 0:
    print("FAILURES:")
    for (an, en, _) in failures: print(an, en)

    for an, en, e in failures:
        print(e)