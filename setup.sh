#!/bin/bash

system=${1-default}

if [ $system == arc ]
then
    echo "Installing dependencies for ARC..."
    module purge
    module load mujoco/2.00
    module load python/anaconda3/2019.03
    conda create --prefix $HOME/lmorl/venv --copy python=3.6
    conda activate $HOME/lmorl/venv 
else
    echo "Installing dependencies..."
    virtualenv venv
    source venv/bin/activate
fi

pip install -r requirements.txt
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip install -e .
cd ..

if [ $system == arc ]
then
    conda deactivate
else
    deactivate
fi