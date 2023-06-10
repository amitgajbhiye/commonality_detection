#!/bin/bash --login

#SBATCH --job-name=getClaSimWords

#SBATCH --output=logs/classificatin_vocabs/out_get_classification_vocabs_similar_words.txt
#SBATCH --error=logs/classificatin_vocabs/err_get_classification_vocabs_similar_words.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute
#SBATCH --mem=12G

#SBATCH -t 0-02:00:00

conda activate venv

python3 commonality/com_det.py classi_vocab

echo 'Job Finished !!!'