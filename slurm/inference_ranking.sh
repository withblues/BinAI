#!/bin/bash
#SBATCH --partition=NvidiaAll
#SBATCH --output=logs/ranking_%j.out

python precompute_inference.py --data_dir outputs --output_dir home_outputs --model ranking --batch_size 32
