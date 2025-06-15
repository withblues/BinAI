#!/bin/bash
#SBATCH --partition=NvidiaAll

python python distillation.py --output_dir outputs --data_dir outputs
