#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=SELD
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=spherev3/slurm_output_%A_%a.out
#SBATCH --array=0

cd ~/phd/Embisonics/seld-dcase2023-baseline
export HYDRA_FULL_ERROR=1

module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-trainer

python3 batch_feature_extraction.py 2 
python3 train_seldnet.py 2