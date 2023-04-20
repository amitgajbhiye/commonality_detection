#!/bin/bash --login

#SBATCH --job-name=getEmb

#SBATCH --output=logs/out_get_nearest_negighbours_ufet_wiki.txt
#SBATCH --error=logs/err_get_nearest_negighbours_ufet_wiki.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G

#SBATCH -t 0-02:00:00

conda activate venv

python3 gensim_modelling.py

echo 'Job Finished !!!'