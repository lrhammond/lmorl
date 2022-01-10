import argparse

from src.train import train_from_params
from tqdm import tqdm
from datetime import datetime
import os
import src.constants as constants
from src.batch_definitions import *


def run_test_group(test_group_label: str, test_group_params: list):
    successes = []
    failures = []

    session_pref = os.path.join(constants.data_dir, test_group_label,
                                datetime.now().strftime('%Y%m%d-%H%M%S'))

    pbar = tqdm(test_group_params)
    for train_params in pbar:
        pbar.set_description(f"{len(successes)}S:{len(failures)}F")
        try:
            train_from_params(train_params=train_params,
                              session_pref=session_pref)
            successes.append(train_params)
        except Exception as e:
            failures.append((train_params, e))

    print(f"Final total: {len(successes)}S:{len(failures)}F")

    if len(failures) > 0:
        print("FAILURES:")
        for tps, e in failures:
            tps.render_and_print()
            print(e)

    print("Run this to see results:")
    print(f"cd ~/lmorl && source venv/bin/activate && tensorboard --logdir={session_pref}")


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
