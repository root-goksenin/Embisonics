#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=GRAMTi
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/embisonics_icassp/Embisonics

module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-trainer

python3 train_new.py data=visage_ambisonics
