#!/bin/bash --login

#SBATCH --job-name=getClusters

#SBATCH --output=logs/out_get_clusters_w2v_numberbatch_fasttext.txt
#SBATCH --error=logs/err_get_clusters_w2v_numberbatch_fasttext.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=40G

#SBATCH -t 3-00:00:00

##SBATCH --gres=gpu:1

conda activate venv

python3 commonality/com_det.py

echo 'Job Finished !!!'