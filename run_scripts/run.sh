#!/bin/bash --login

#SBATCH --job-name=comDet

#SBATCH --output=logs/out_all_com_det_ufet_wiki_words.txt
#SBATCH --error=logs/err_all_com_det_ufet_wiki_words.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=20G

#SBATCH -t 0-03:00:00

##SBATCH --gres=gpu:1

conda activate venv

python3 commonality/gensim_modelling.py

echo 'Job Finished !!!'