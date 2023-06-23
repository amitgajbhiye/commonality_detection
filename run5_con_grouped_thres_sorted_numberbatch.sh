#!/bin/bash --login

#SBATCH --job-name=ConGroupThreshSorted

#SBATCH --output=logs/out_con_grouped_thres_sorted_numberbatch.txt
#SBATCH --error=logs/err_con_grouped_thres_sorted_numberbatch.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem

#SBATCH --mem=20G
#SBATCH -t 0-05:00:00

conda activate venv

python3 get_sorted_df.py

echo 'Job Finished !!!'