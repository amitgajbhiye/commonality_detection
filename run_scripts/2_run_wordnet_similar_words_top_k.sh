#!/bin/bash --login

#SBATCH --job-name=cWordNet

#SBATCH --output=logs/classification_vocabs/out_get_wordnet_similar_words.txt
#SBATCH --error=logs/classification_vocabs/err_get_wordnet_similar_words.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH --mem=40G

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu7d


conda activate venv

python3 commonality/com_det_topk_similar.py classi_vocab wordnet

echo 'Job Finished !!!'