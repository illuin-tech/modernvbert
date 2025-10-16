#!/bin/bash
#SBATCH -A nwd@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --job-name=post_train
#SBATCH --mem-bind=local
#SBATCH --cpus-per-task=64
#SBATCH --hint=nomultithread

module purge
module load arch/h100 cuda/12.8.0
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/datasets
export HF_HOME=$SCRATCH/colpali_data/.cache/huggingface
# export PYTHONPATH=/lustre/fswork/projects/rech/qts/ucg53vj/colpali_data/venv_doria/lib/python3.10/site-packages:PYTHONPATH
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_PROJECT="vbert"

source .venv/bin/activate

wandb offline

accelerate launch train.py -c "/linkhome/rech/genrce01/unj98sd/visual_encoder/posttraining/ablations/contrastive_forcing/config/contrastive_mixed_modality_250k.yml"
accelerate launch evaluate.py -c "/linkhome/rech/genrce01/unj98sd/visual_encoder/posttraining/ablations/contrastive_forcing/config/contrastive_mixed_modality_250k.yml"
