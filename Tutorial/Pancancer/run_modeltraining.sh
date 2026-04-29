#!/bin/bash
#SBATCH --job-name=SpiderNet_SimMoreMI
#SBATCH --time=2-00:00:00
#SBATCH -c 6
#SBATCH --mem=32Gb
#SBATCH -p gpu-large
#SBATCH --gres=gpu:A6000:1
#SEATCH --pty bash
#SBATCH --output=SpiderNet_Simgeneration_moreMI.out
#SBATCH --error=SpiderNet_Simgeneration_moreMI.err

source activate base
conda activate SpiderNet_env
export CUDA_LAUNCH_BLOCKING=1
#
python -u SimulationData_generation_moreMI.py