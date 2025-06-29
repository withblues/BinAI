#!/bin/bash
#SBATCH --partition=NvidiaAll

#python distillation.py --output_dir outputs --data_dir outputs

#python ranking.py --data_dir outputs --output_dir outputs
python train.py --data_dir outputs --output_dir outputs --mode distil