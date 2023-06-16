#!/bin/bash --login

#SBATCH --job-name=RP_BN_1Hot

#SBATCH --output=logs/classification_vocabs/out_get_one_hot_enc_fasttext_babelnet_clusters.txt
#SBATCH --error=logs/classification_vocabs/err_get_one_hot_enc_fasttext_babelnet_clusters.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute,highmem

#SBATCH --mem=100G
#SBATCH -t 0-20:00:00

conda activate venv

python3 commonality/one_hot_converter.py output_files/classification_vocabs_similar_thresh_50/bablenet_domain output_files/classification_vocabs_similar_thresh_50/bablenet_domain/onehot_encodings

echo 'Job Finished !!!'