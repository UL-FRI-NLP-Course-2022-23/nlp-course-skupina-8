#!/bin/bash
#SBATCH --nodes=1           # run on the same node
#SBATCH --ntasks=1          # run one time
#SBATCH --cpus-per-task=12  # with 12 threads
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=2-00:00
#SBATCH --job-name="nlp"
#SBATCH --mem=32G
#SBATCH --output=logs/nlp-%J.out
#SBATCH --error=logs/nlp-%J.err

srun singularity exec --nv /d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif python3 train.py
