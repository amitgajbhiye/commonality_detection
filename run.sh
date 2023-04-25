#!/bin/bash --login

#SBATCH --job-name=comDet

#SBATCH --output=logs/out_grid_com_det_ufet_wiki_words.txt
#SBATCH --error=logs/err_grid_com_det_ufet_wiki_words.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=20G

#SBATCH -t 3-00:00:00

##SBATCH --gres=gpu:1

conda activate venv

python3 commonality/gensim_modelling.py

echo 'Job Finished !!!'