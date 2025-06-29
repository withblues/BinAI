#!/bin/bash
#SBATCH --partition=Krater


python tokenize_data.py --data_dir outputs --output_dir outputs

#python tokenize_data.py --data_dir outputs --output_dir outputs --split valid
