#!/bin/bash
#SBATCH --nodes=1           # run on the same node
#SBATCH --ntasks=1          # run one time
#SBATCH --cpus-per-task=8   # with 8 threads
#SBATCH --partition gpu
#SBATCH --gpus-per-node=0
#SBATCH --time=0-00:25
#SBATCH --job-name="nlp"
#SBATCH --mem=10000
#SBATCH --output=logs/nlp-%J.out
#SBATCH --error=logs/nlp-%J.err

# #SBATCH --gres=gpu:1        #--gpus=1

# exec container
srun -n1 -N1 -c8 --preserve-env --pty --partition=GPU singularity exec /d/hpc/projects/FRI/tp1859/nlp_project8/lma/hf.sif python3 -c "print('hello world')"

# nvidia-smi
