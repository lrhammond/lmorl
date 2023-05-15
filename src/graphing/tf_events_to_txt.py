from tensorflow.python.summary.summary_iterator import summary_iterator
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def convert_tf_to_txt(p):
    assert (os.path.exists(p)), f"path {p} does not exist"
    s_iter = summary_iterator(p)
    rewards = []
    costs = []
    for line in s_iter:
        for val in line.summary.value:
            if "Reward" in val.tag: rewards.append(val.simple_value)
            if "Cost" in val.tag: costs.append(val.simple_value)
    assert (len(rewards) == len(costs))
    mid_name = os.path.split(p)[0].replace("data/", "")
    file_name = os.path.split(p)[1].replace("events.out.tfevents.", "") + ".txt"
    dir_name = os.path.join("data", "txts", mid_name)
    os.makedirs(dir_name, exist_ok=True)
    txt_p = os.path.join(dir_name, file_name)
    assert (not os.path.exists(txt_p)), f"path {txt_p} already exists"
    with open(txt_p, 'w') as f:
        for i, (r, c) in enumerate(zip(rewards, costs)):
            f.write('{},{}\n'.format(r, c))


if __name__ == "__main__":
    paths = []
    for path, b, file_names in os.walk("data"):
        for file_name in file_names:
            if "tfevents" in file_name:
                paths.append(os.path.join(path, file_name))

    for file_path in tqdm(paths):
        try:
            convert_tf_to_txt(file_path)
        except Exception as e:
            print(e)
