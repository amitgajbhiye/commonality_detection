#!/bin/bash --login

#SBATCH --job-name=FTembDICT

#SBATCH --output=logs/out_get_fasttext_embedding_dict.txt
#SBATCH --error=logs/err_get_fasttext_embedding_dict.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=20G

#SBATCH -t 0-03:00:00

##SBATCH --gres=gpu:1

conda activate venv

python3 commonality/com_det.py

echo 'Job Finished !!!'