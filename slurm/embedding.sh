#!/bin/bash
#SBATCH --partition=NvidiaAll

python precompute.py --data_dir outputs/baseline-train-indexed.pkl --output_dir outputs/clap-train-embeddings.pkl

#python precompute.py --data_dir outputs/baseline-valid-indexed.pkl --output_dir outputs/clap-valid-embeddings.pkl