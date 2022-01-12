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
    "VaR_AC",
    "seqLA2C",
    "seqLPPO",
    "LA2C2nd",
    "seqLA2C2nd",
    "LPPO2nd",
    "seqLPPO2nd"
]

lex_agents = [
    "LDQN",
    "LA2C",
    "LPPO",
    "seqLA2C",
    "seqLPPO",
    "LA2C2nd",
    "seqLA2C2nd",
    "LPPO2nd",
    "seqLPPO2nd"
]


# TODO - add cont list, add checks for agent-env compat
agent_names = disc_agent_names + []

env_names = [
    "Bandit",
    "CartSafe",
    "GridNav"
]
