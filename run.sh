#!/bin/bash --login

#SBATCH --job-name=getEmb

#SBATCH --output=logs/out_get_nearest_negighbours_ufet_wiki.txt
#SBATCH --error=logs/err_get_nearest_negighbours_ufet_wiki.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=40G

#SBATCH -t 0-02:30:00

##SBATCH --gres=gpu:1

conda activate venv

python3 commonality/gensim_modelling.py

echo 'Job Finished !!!'