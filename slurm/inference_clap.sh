#!/bin/bash
#SBATCH --partition=NvidiaAll
#SBATCH --output=logs/clap_%j.out

python precompute_inference.py --data_dir outputs --output_dir home_outputs --model clap --batch_size 32