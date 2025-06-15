#!/bin/bash
#SBATCH --partition=Krater

python analysis.py --data_dir outputs --split train
