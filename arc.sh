#!/bin/bash

echo "Installing dependencies for ARC..."

module purge
module load anaconda3/2018.12
module load mujoco/2.00

conda create --prefix $HOME/lmorl/venv --copy python=3.6
conda init bash
conda activate $HOME/lmorl/venv 

pip install -r requirements.txt

conda deactivate

mkdir results logs