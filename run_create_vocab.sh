#!/bin/bash --login

#SBATCH --job-name=getVOcab

#SBATCH --output=logs/out_get_word_counts_100K_en_wikipedia.txt
#SBATCH --error=logs/err_get_word_counts_100K_en_wikipedia.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
##SBATCH --gres=gpu:1

#SBATCH --mem=10G

#SBATCH -t 0-03:00:00

conda activate venv

python3 commonality/vocab.py

echo 'Job Finished !!!'