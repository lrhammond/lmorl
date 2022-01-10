#!/bin/bash
#SBATCH --partition=htc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=1:00:00

module purge
module load anaconda3/2018.12
module load mujoco/2.00

PROJ_DIR=$HOME/PycharmProjects/lmorl/
VENV_PATH=$PROJ_DIR/venv
TEST_LOG_PATH=$PROJ_DIR/logs/test.log
if [ ! -d $PROJ_DIR ]; then
  echo "Project path ($VENV_PATH) not found, please manually specify above"
fi
if [ ! -d $VENV_PATH ]; then
  echo "venv ($VENV_PATH) not found, please manually specify below"
fi
if [ ! -d $TEST_LOG_PATH ]; then
  touch $TEST_LOG_PATH
fi
source activate $VENV_PATH
python $PROJ_DIR/src/test.py  &> $PROJ_DIR/logs/test.log
source deactivate

##SBATCH --partition=htc
##SBATCH --gres=gpu:1
##SBATCH --nodes=1
##SBATCH --time=1:00:00
#
#module purge
#module load anaconda3/2018.12
#module load mujoco/2.00
#
#PROJ_DIR=$HOME/PycharmProjects/lmorl/
#VENV_PATH=$PROJ_DIR/venv
#TEST_LOG_PATH=$PROJ_DIR/logs/test.log
#if [ ! -d $PROJ_DIR ]; then
#  echo "Project path ($VENV_PATH) not found, please manually specify above"
#fi
#if [ ! -d $VENV_PATH ]; then
#  echo "venv ($VENV_PATH) not found, please manually specify below"
#fi
#if [ ! -d $TEST_LOG_PATH ]; then
#  touch $TEST_LOG_PATH
#fi
#source activate $VENV_PATH
#python $PROJ_DIR/src/test.py  &> $PROJ_DIR/logs/test.log
#source deactivate
