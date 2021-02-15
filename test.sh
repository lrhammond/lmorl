#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=1:00:00

module purge
module load anaconda3/2018.12
module load mujoco/2.00

source activate $HOME/lmorl/venv
python $HOME/lmorl/src/test.py  &> $HOME/lmorl/logs/test.log
source deactivate