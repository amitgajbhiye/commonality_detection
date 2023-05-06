#!/bin/bash --login

#SBATCH --job-name=getClusters

#SBATCH --output=logs/out_dummy.txt
#SBATCH --error=logs/err_dummy.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=30G

#SBATCH -t 3-00:00:00

##SBATCH --gres=gpu:1

conda activate venv

python3 commonality/com_det.py

echo 'Job Finished !!!'