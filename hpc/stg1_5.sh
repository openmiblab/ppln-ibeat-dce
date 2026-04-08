#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=stg1_5
#SBATCH --output=logs/stg1_5.out
#SBATCH --error=logs/stg1_5.err

module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0

# Prevent pop-up errors on headless HPC nodes
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

ENV="/mnt/parscratch/users/$(whoami)/envs/dce"
CODE="/mnt/parscratch/users/$(whoami)/ppln-ibeat-dce/src/ibeat_dce"
DATA_DIR="/mnt/parscratch/users/$(whoami)/data/ibeat_dce/stage_1_download"

# Run Stage 1.5, passing the HPC data directory as an argument
srun "$ENV/bin/python" "$CODE/stage_1_5_unzip.py" "$DATA_DIR"