import argparse

from src.train import train_from_params
from tqdm import tqdm
from datetime import datetime
import os
import src.constants as constants
from src.batch_definitions import *


class SummaryWriter:
    pass


def run_test_group(test_group_label: str, test_group_params: list):
    successes = []
    failures = []

    session_pref = os.path.join(constants.data_dir, test_group_label,
                                datetime.now().strftime('%Y%m%d-%H%M%S'))

    # Create a dummy writer to bait out tensorflow warnings before tqdm
    SummaryWriter()
    print()
    print("Run this to see results:")
    print(f"cd ~/lmorl && source venv/bin/activate && tensorboard --logdir={session_pref}")
    print()

    if len(test_group_params) >= 2:
        pbar = tqdm(test_group_params, colour="blue")
    else:
        pbar = test_group_params

    for train_params in pbar:
        try:
            train_from_params(train_params=train_params,
                              session_pref=session_pref)
            successes.append(train_params)
        except Exception as e:
            failures.append((train_params, e))

    print(f"Final total: {len(successes)}S:{len(failures)}F")

    if len(failures) == 1:
        print("FAILURE:")
        print(failures[0][0])
        raise failures[0][1]
    elif len(failures) >= 2:
        print("FAILURES:")
        raise Exception(failures)

    print()
    print("Run this to see results:")
    print(f"cd ~/lmorl && source venv/bin/activate && tensorboard --logdir={session_pref}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests")
    test_group_names = list(batch_definitions.keys())
    parser.add_argument("test_group_name",
                        type=str,
                        default=test_group_names[0],
                        choices=test_group_names,
                        help=f"Choose a group of tests defined in tests/test_groups.py.")
    args = parser.parse_args()
    test_group_name = args.test_group_name
    test_group_params = batch_definitions[test_group_name]
    run_test_group(test_group_name, test_group_params)
