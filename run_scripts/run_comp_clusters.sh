#!/bin/bash --login

#SBATCH --job-name=CompCluster

#SBATCH --output=logs/out_create_complimentary_clusters_relbert_filetered_data_w2v_numberbatch_fasttext.txt
#SBATCH --error=logs/err_create_complimentary_clusters_relbert_filetered_data_w2v_numberbatch_fasttext.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute
#SBATCH --mem=20G

#SBATCH -t 2-00:00:00

conda activate venv

python3 commonality/get_complementary_cluster.py

echo 'Job Finished !!!'