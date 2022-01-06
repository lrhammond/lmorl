from src.train import TrainingParameters, train_from_params
from src.envs import env_names
from src.make_agent import agent_names
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt

# Draw a subset for faster testing
agent_names = agent_names[1:3]
env_names = env_names[1:]

pairs = [(agent_name, env_name) for agent_name in agent_names for env_name in env_names]
successes = []
failures = []

train_logs = {(agent_name, env_name): None for agent_name, env_name in pairs}
tb_log_path_base = f"./runs/learn_tests/{datetime.now().strftime('%Y%m%d-%H%M%S')}"

print("Running learning tests")
pbar = tqdm(pairs)
for agent_name, env_name in pbar:
    pbar.set_description(f"{len(successes)}S:{len(failures)}F")
    # TODO address interacts
    tb_log_path = os.path.join(tb_log_path_base, f"{env_name}-{agent_name}")
    train_params = TrainingParameters(agent_name=agent_name, env_name=env_name,
                                      interacts=10, save_location="sanity_test_logs",
                                      tb_log_path=tb_log_path)
    try:
        train_log = train_from_params(train_params)
        train_logs[(agent_name, env_name)] = train_log
        successes.append((agent_name, env_name))
    except Exception as e:
        failures.append((agent_name, env_name, e))


print(f"Final total: {len(successes)}S:{len(failures)}F")
print("FAILURES:")
for (agent_name, env_name, _) in failures: print(agent_name, env_name)
for agent_name, env_name, e in failures:
    print(e)

# fig, axs = plt.subplots(len(env_names), 2)
# fig.suptitle('Environments')
#
# for (j, title) in enumerate(["Rewards", "Costs"]):
#     for (i, env_name) in enumerate(env_names):
#         axs[i, j].set_title(title + " " + env_name)
#         for agent_name in agent_names:
#             log = train_logs[(agent_name, env_name)]
#             if log is None: data=[0]
#             else:
#                 if title == "Rewards":
#                     data = [r for (_, r, _ ) in log]
#                 elif title == "Costs":
#                     data = [c for (_, c, _) in log]
#                 else: raise Exception("Rewards or Costs only not " + title)
#             axs[i, j].plot(data, label=agent_name)
#         axs[i, j].legend()
# plt.show()
