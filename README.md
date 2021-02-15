# Lexicographic Multi-Objective Reinforcement Learning

## Setup

Note that the instructions below require a valid [MuJoCo](http://www.mujoco.org/) installation and license due to the use of [OpenAI Safety Gym](https://github.com/openai/safety-gym).

### Default

Begin by cloning and entering the repository:

```
git clone https://github.com/lrhammond/lmorl.git
cd lmorl
```

Then, set up your Python environment to taste (for example, by using `virtualenv`, or `conda`). The following instructions are given using `pip`, though your preferred package manager may vary. Install the required packages using:

```
pip install -r requirements.txt
```

Next, download and install the OpenAI Safety Gym repository as a sub-directory of `lmorl` (note that this step assumes a valid installation of MuJoCo at the `$HOME/.mujoco/mujoco200` and a valid license key at `$HOME/.mujoco/mjkey.txt`):

```
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip install -e .
cd ..
```

Finally, create two extra sub-directories of `lmorl` (that are used for recording and logging data from experiments) by running:

```
mkdir results logs
```

### ARC

If using [ARC](https://www.arc.ox.ac.uk/) then first navigate to your home directory (use `cd $HOME` on `arcus-htc`). Begin by cloning and entering the repository:

```
git clone https://github.com/lrhammond/lmorl.git
cd lmorl
```

The remaining setup steps are automated, but assumes that there is a valid MuJoCo license located at `$HOME/.mujoco/mjkey.txt`. Finish the setup by running:

```
sh -i arc.sh
``` 

This will activate the relevant ARC modules, create a conda environment `venv` with all the required packages, and create any extra required sub-directories. It will also install the OpenAI Safety Gym repository as a sub-directory of `lmorl` and add the package to `venv`.

## Running Experiments

### Default

Individual experiments may be run using:

```
python src/main.py <agent_name> <robot> <task> <difficulty> <episodes> <iteration>
```

### ARC

Individual experiments may be submitted to ARC using the command:

```
sbatch experiment.sh <agent_name> <robot> <task> <difficulty> <episodes> <iteration>
```

To run all experiments, use `sh run_all.sh` instead.

## Questions?

Please send questions/feedback to [Joar](mailto:joar.skalse@cs.ox.ac.uk) or [Lewis](mailto:lewis.hammond@cs.ox.ac.uk).
