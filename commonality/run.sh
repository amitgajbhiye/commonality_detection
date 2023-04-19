#!/bin/bash --login

#SBATCH --job-name=getEmb

#SBATCH --output=logs/out_get_relbert_embeds_ufet_concepts_netp_props.txt
#SBATCH --error=logs/err_get_relbert_embeds_ufet_concepts_netp_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=16G

#SBATCH -t 0-00:30:00

conda activate venv

python3 gensim_modelling.py

echo 'Job Finished !!!'