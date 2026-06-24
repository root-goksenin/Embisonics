#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=MWMAE
#SBATCH --ntasks=1
#SBATCH --exclude=gcn118
#SBATCH --time=00:01:00
#SBATCH --output=localization_hear_mic_inv/slurm_output_%A_%a.out
#SBATCH --array=0


cd ~/phd/Embisonics
module load 2023
module load Anaconda3/2023.07-2
source activate spatial-ssast-eval
cd listen-eval-kit


embeddings_dir=/projects/0/prjs1338/LocalizationEmbeddingsSPHERE_v2_fr
score_dir=try_real_world
tasks_dir=/projects/0/prjs1338/ears

weights=/gpfs/work4/0/prjs1338/embisonics_mae_saved_try_equiv_no_gate_fr/InChannels=7/Fraction=1.0/CleanDataFraction=0.0/Model=GRAM-T/ModelSize=base/LR=0.0004/BatchSize=16/NrSamples=1/Patching=frame/InputL=1024/step=30000.ckpt
model_name=hear_configs.SPHERE_v2
strategy=raw
model_options="{\"strategy\": \"$strategy\"}"
task_name=ears-v1.0.0-16-spherical

# embeddings_dir=/projects/0/prjs1338/LocalizationEmbeddingsSPHERE_v2_fr
# score_dir=try_real_world
# tasks_dir=/projects/0/prjs1338/realseld/RealSELD

# weights=/gpfs/work4/0/prjs1338/embisonics_mae_saved_try_equiv_no_gate_fr/InChannels=7/Fraction=1.0/CleanDataFraction=0.0/Model=GRAM-T/ModelSize=base/LR=0.0004/BatchSize=16/NrSamples=1/Patching=frame/InputL=1024/step=40000.ckpt
# model_name=hear_configs.SPHERE_v2
# strategy=raw
# model_options="{\"strategy\": \"$strategy\"}"
# task_name=tau2019-v1.0.0-full


python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir --model-options "$model_options" --model $weights 
python3 -m heareval.predictions.runner $embeddings_dir/$model_name-strategy=$strategy/$task_name

mkdir -p /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy/$task_name

mv $embeddings_dir/$model_name-strategy=$strategy/$task_name/test.predicted-scores-localization.json /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy/$task_name
mv $embeddings_dir/$model_name-strategy=$strategy/$task_name/*.pkl /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy/$task_name
mv $embeddings_dir/$model_name-strategy=$strategy/$task_name/*embeddings.npy /projects/0/prjs1338/$score_dir/$model_name-strategy=$strategy/$task_name

rm -r -d -f $embeddings_dir/$model_name-strategy=$strategy/$task_name
