#!/bin/bash
#SBATCH -p #######-gpu
#SBATCH --account=#######
#SBATCH --job-name=#######
#SBATCH -o logs_cdisn_training_%A_%a_#######.out
#SBATCH -e logs_cdisn_training_%A_%a_#######.err
#SBATCH --array=1-foo
#SBATCH -c 2
#SBATCH --mem-per-cpu=8G
#SBATCH --exclusive
#SBATCH --gres=gpu:1

module load Python-GPU/3.7.6

PYTHONPATH=foo/cross-domain-learning python cdisn_training_foo.py -n $SLURM_CPUS_PER_TASK
