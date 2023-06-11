#!/bin/bash --login

#SBATCH --job-name=ExSumo

#SBATCH --output=logs/ontology_completion/out_dummy_all_except_sumo_similar_words.txt
#SBATCH --error=logs/ontology_completion/err_dummy_all_except_sumo_similar_words.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH --mem=40G
#SBATCH -p compute
#SBATCH -t 0-02:00:00

conda activate venv

python3 commonality/com_det_topk_similar.py ontology_comp all_except_sumo

echo 'Job Finished !!!'