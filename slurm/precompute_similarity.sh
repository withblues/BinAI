#!/bin/bash
#SBATCH --partition=Krater

python precompute_similarity.py --data_dir outputs --output_dir outputs

#python precompute_similarity.py --data_dir outputs --output_dir outputs --split valid
