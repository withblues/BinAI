#!/bin/bash
#SBATCH --partition=NvidiaAll
#SBATCH --output=logs/distil_%j.out

python precompute_inference.py --data_dir outputs --output_dir home_outputs --model distil --batch_size 32