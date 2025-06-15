#!/bin/bash
#SBATCH --partition=NvidiaAll

python distillation.py --output_dir outputs --data_dir outputs
