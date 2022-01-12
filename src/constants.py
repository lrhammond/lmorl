data_dir = "./data"

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

# TODO - add cont list, add checks for agent-env compat
agent_names = disc_agent_names + []

env_names = [
    "Bandit",
    "CartSafe",
    "GridNav"
]
