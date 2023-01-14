#!/bin/bash
#SBATCH -p scavenger
#SBATCH --account=carlsonlab
#SBATCH --job-name=zResampTS100_12152021
#SBATCH -o logs_resampTS100_%A_%a_12152021.out
#SBATCH -e logs_resampTS100_%A_%a_12152021.err
#SBATCH --array=1
#SBATCH -c 2
#SBATCH --mem-per-cpu=8G
#SBATCH --exclusive

module load Python-GPU/3.7.6

PYTHONPATH=/hpc/home/zcb2/cross-domain-learning python store_smaller_version_of_cached_subset.py -n $SLURM_CPUS_PER_TASK
