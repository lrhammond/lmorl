#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=72:00:00

module purge
module load anaconda3/2018.12
module load mujoco/2.00

source activate $HOME/lmorl/venv
python $HOME/lmorl/src/run.py $1 $2 $3 $4 $5 $6 &> $HOME/lmorl/logs/$1-$2-$3-$4-$5-$6.txt
source deactivate