# Lexicographic Multi-Objective Reinforcement Learning

## Setup

First clone the repository using `git clone https://github.com/lrhammond/lmorl.git` and enter it using `cd lmorl`. Note that the instructions below require a valid [MuJoCo](http://www.mujoco.org/) installation and license due to the use of [OpenAI Safety Gym](https://github.com/openai/safety-gym).

### Default

First, set up your Python environment to taste (for example, by using `virtualenv`, or `conda`). The following instructions are given using `pip`, though your preferred package manager may vary. Begin by installing the required packages:

```pip install -r requirements.txt```

Then proceed to download and install the OpenAI Safety Gym repository as a sub-directory of `lmorl`:

```
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip install -e .
cd ..
```

Finally, create two extra sub-directories of `lmorl` (that are used for recording and logging data from experiments) by running `mkdir results logs`.

### ARC

If using [ARC](https://www.arc.ox.ac.uk/) then run `sh -i arc.sh` (note that this setup assumes that the `lmorl` repo is located at `$HOME/lmorl`). This will activate the relevant modules, create a virtual environment `venv` with all the required packages, and create any extra required sub-directories. It will also install the OpenAI Safety Gym repository as a sub-directory of `lmorl` and add the package to `venv`.

## Running Experiments

### Default

Individual experiments may be run using:

```python src/main.py <agent_name> <robot> <task> <difficulty> <episodes> <iteration>```

### ARC

Individual experiments may be submitted to ARC using the command:

```sbatch experiment.sh <agent_name> <robot> <task> <difficulty> <episodes> <iteration>```

To run all experiments use `sh run_all.sh`.

## Questions?

Please send questions/feedback to [Joar](mailto:joar.skalse@cs.ox.ac.uk) or [Lewis](mailto:lewis.hammond@cs.ox.ac.uk).
