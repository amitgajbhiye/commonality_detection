#!/bin/bash --login

#SBATCH --job-name=cBabNet

#SBATCH --output=logs/classification_vocabs/out_get_babelnet_similar_words.txt
#SBATCH --error=logs/classification_vocabs/err_get_babelnet_similar_words.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH --mem=40G

#SBATCH --mem=40G
#SBATCH -p compute
#SBATCH -t 3-00:00:00

conda activate venv

python3 commonality/com_det_topk_similar.py classi_vocab babelnet

echo 'Job Finished !!!'