#!/bin/bash

module load mujoco/2.00
module load python/anaconda3/2019.03
conda create --prefix venv --copy python=3.6
source activate venv

pip install -r requirements.txt
git clone https://github.com/openai/safety-gym.git
cd safety-gym
pip install -e .
cd ..

conda deactivate