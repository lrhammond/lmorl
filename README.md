# Lexicographic Multi-Objective Reinforcement Learning

## Installation

To view installation instructions, go to [INSTALLATION.md](https://github.com/lrhammond/lmorl/blob/main/INSTALLATION.md)

## Running Experiments

### Default
To test your installation, run:
``` commandline
python -m src.run_batches humble_batch
```

Some individual experiments may be run using:
``` commandline
python -m src.train --agent_name=_ --env_name=_ --num_episodes=_
```

### Running Batches
To simplify training and running experiments, we use a "batch" system. 
The class `TrainingParameters` specifies all of the information needed to train a single agent on a particular environment.
The file `batch_definitions` contains "batches": lists of instances of `TrainingParameter` used to specify a sequence of training runs.
For example, the following batch would compare `DQN` and `AC` in the `CartSafe` environment:
``` python
"DQN_vs_AC": [
    TrainingParameters( agent_name="DQN", env_name="CartSafe"),
    TrainingParameters( agent_name="AC", env_name="CartSafe"),    
],
```

You can run these predefined batches using the `run_batches.py file`:
```
python -m src.run_batches DQN_vs_AC
```

## Project structure
As above, `run_batches.py` and `batch_definitions.py` define meta-level code for managing experiments.
The file `train.py` run individual experiments and controls the state-agent interaction loop.
Individual agents are defined in the `agents` directory and environments in the `envs.py` file.
The `graphing` directory contains code for generating the graphs included in the paper.

## Questions?

Please send questions/feedback to [Joar](mailto:joar.skalse@cs.ox.ac.uk) or [Lewis](mailto:lewis.hammond@cs.ox.ac.uk).