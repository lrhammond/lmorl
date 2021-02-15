#!/bin/bash

echo "Installing dependencies for ARC..."

# Load required ARC modules
module purge
module load anaconda3/2018.12
module load mujoco/2.00

# Create new virtual environment
conda create --prefix $HOME/lmorl/venv --copy python=3.7
source activate $HOME/lmorl/venv 

# Install packages
pip install -r requirements.txt

# Note that, apparently, safety-gym requires numpy~=1.17.4 but mujoco-py requires a more recent version
pip uninstall numpy
pip install numpy==1.20.1

source deactivate

# Create extra directories for storing experimental output
mkdir results logs