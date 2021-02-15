#!/bin/bash

echo "Installing dependencies for ARC..."

mkdir results
mkdir logs

module purge
module load anaconda3/2018.12
module load mujoco/2.00

conda create --prefix $HOME/lmorl/venv --copy python=3.6
conda init bash
conda activate $HOME/lmorl/venv 

pip install -r requirements.txt
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip install -e .
cd ..

conda deactivate