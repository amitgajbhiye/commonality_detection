#!/bin/bash --login

#SBATCH --job-name=oSumo

#SBATCH --output=logs/ontology_completion/out_get_sumo_similar_words.txt
#SBATCH --error=logs/ontology_completion/err_get_sumo_similar_words.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH --mem=60G

#SBATCH -p gpu,gpu_v100
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu7d


conda activate venv

python3 commonality/com_det_topk_similar.py ontology_comp sumo

echo 'Job Finished !!!'