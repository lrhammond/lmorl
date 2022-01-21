# Installation Steps
Below are the steps taken to get tests running on a fresh Ubuntu docker container.

## Basic Requirements
Ensure packages are up-to-date.
``` bash
sudo apt-get install update
sudo apt-get install upgrade
```

Install package and version control.
``` bash
sudo apt-get install git
sudo apt-get install python3
sudo apt-get install pip
python3 -m pip install --user --upgrade pip
```

Install venv for creating a virtual environment for this project.
``` bash
python3 -m pip install --user virtualenv
sudo apt install pythib3.8-venv
```

## Setting up lmorl
Clone the repo with `git clone https://github.com/lrhammond/lmorl.git`. 
Then create a virtual environment for this project:
``` bash
cd lmorl/
python3 -m venv venv
source venv/bin/activate
```

Install tensorboard to get a friendly GUI for managing experiments:
``` bash
sudo apt-get install tensorboard
```

Manually install some dependencies. This avoided some conflicts for me.
```
sudo apt install gcc & pip install mujoco_py==2.0.2.8
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

Finally, install the remaining dependencies.
``` bash
pip install -r requirements.txt
```

## Verify installation
Check your installation by running the simplest test: `humble_test`.
```
python3 -m src.run_batches humble_test
cd ~/lmorl && source venv/bin/activate && tensorboard --logdir=./data/humble_test/
```

To do a more thorough check, run `exit_tests`:
```
python3 -m src.run_batches exit_tests
cd ~/lmorl && source venv/bin/activate && tensorboard --logdir=./data/humble_test/
```