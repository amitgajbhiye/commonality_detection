#!/bin/bash --login

#SBATCH --job-name=CompCluster

#SBATCH --output=logs/out_create_complimentary_clusters_relbert_filetered_data.txt
#SBATCH --error=logs/err_create_complimentary_clusters_relbert_filetered_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=12G

#SBATCH -t 0-02:00:00


conda activate venv

python3 commonality/complete_cluster.py

echo 'Job Finished !!!'