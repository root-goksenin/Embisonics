#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=GRAMTi
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_output_%A_%a.out
#SBATCH --array=0
#SBATCH --constraint=scratch-node


cd ~/embisonics_icassp/Embisonics

module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-trainer

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

rclone copy /projects/0/prjs1261/visage/audios/audio_wds $TMPDIR/ --include "*.tar" --transfers $(nproc) --checkers $(nproc) -L

python3 train.py \
    "data.glob='${TMPDIR}/shard-{000000..000048}.tar'"
