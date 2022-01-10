from src.train import TrainingParameters, train_from_params
from src.envs import env_names
from src.make_agent import disc_agent_names
from tqdm import tqdm
from datetime import datetime
import os

pairs = [(agent_name, env_name) for agent_name in disc_agent_names for env_name in ["Bandit"]]
successes = []
failures = []

train_logs = {(agent_name, env_name): None for agent_name, env_name in pairs}
base_path = "./runs_tests/agent_tests"
tb_log_path_base = os.path.join(base_path,
                                datetime.now().strftime('%Y%m%d-%H%M%S'))


pbar = tqdm(pairs)
for agent_name, env_name in pbar:
    pbar.set_description(f"{len(successes)}S:{len(failures)}F")

    tb_log_path = os.path.join(tb_log_path_base, f"{env_name}-{agent_name}")
    train_params = TrainingParameters(agent_name=agent_name, env_name=env_name,
                                      num_episodes=100, save_location="sanity_test_logs",
                                      tb_log_path=tb_log_path)
    try:
        train_from_params(train_params)
        successes.append((agent_name, env_name))
    except Exception as e:
        failures.append((agent_name, env_name, e))


print(f"Final total: {len(successes)}S:{len(failures)}F")

if len(failures) > 0:
    print("FAILURES:")
    for (an, en, _) in failures:
        print(an, en)

    for an, en, e in failures:
        print(e)

print("Run this to see results:")
print(f"cd ~/lmorl && source venv/bin/activate && tensorboard --logdir={base_path}")