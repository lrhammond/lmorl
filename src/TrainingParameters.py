import inspect

from src.constants import env_names
from src.constants import agent_names
from dataclasses import dataclass


@dataclass
class TrainingParameters:
    # dataclass attributes created in __init__ and address by self.
    # TODO - perhaps add validation to check when unnecessary parameters are specified e.g. batch_size for tabular
    env_name: str
    agent_name: str
    network: str = "DNN"

    num_episodes: int = None
    num_interacts: int = None

    test_group_label: str = None
    save_every_n: int = None

    buffer_size: int = int(1e4)
    batch_size: int = 128
    update_every: int = 4
    update_every_eps = 32

    epsilon: float = 0.05
    slack: float = 0.05
    lambda_lr_2: float = 0.05
    alpha: float = 1
    beta: float = 0.95

    no_cuda = True

    # After dataclass attributes are initialised, validate the training parameters
    def __post_init__(self):
        assert (self.agent_name in agent_names)
        assert (self.env_name in env_names)
        assert (self.network in ["CNN", "DNN"])
        assert (not (self.num_interacts is None and self.num_episodes is None))
        assert (self.num_interacts is None or self.num_episodes is None)
        if self.num_interacts is not None:
            self.is_interact_mode = True
        else:
            self.is_interact_mode = False

    def render_and_print(self):
        print(self.render_to_string())

    def render_to_string(self):
        x = ""
        for atr_name, atr in inspect.getmembers(self):
            if not atr_name.startswith("_") and not inspect.ismethod(atr):
                x += f" < {atr_name}: {str(atr)} >, "
        return x

    def render_to_file(self, dir):
        x = self.render_to_string()
        with open(dir, "w") as f: f.write(x)