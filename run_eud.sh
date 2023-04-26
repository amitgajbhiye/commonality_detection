#!/bin/bash --login

#SBATCH --job-name=EudSim

#SBATCH --output=logs/out_euclidean_sim.txt
#SBATCH --error=logs/err_euclidean_sim.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=15G

#SBATCH -t 0-02:00:00

##SBATCH --gres=gpu:1

conda activate venv

python3 commonality/similar_words.py

echo 'Job Finished !!!'