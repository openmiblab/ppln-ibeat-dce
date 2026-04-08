#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=stg2
#SBATCH --output=logs/stg2.out
#SBATCH --error=logs/stg2.err

module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

ENV="/mnt/parscratch/users/$(whoami)/envs/dce"
CODE="/mnt/parscratch/users/$(whoami)/ppln-ibeat-dce/src/ibeat_dce"

# Run Stage 2
srun "$ENV/bin/python" "$CODE/stage_2_descriptive.py"

# Sync the extracted descriptive maps and MP4s to your shared drive
rsync -av --no-group --no-perms "/mnt/parscratch/users/$(whoami)/data/ibeat_dce/stage_2_descriptive" "login1:/shared/abdominal_imaging/Shared/ibeat_dce/data/ibeat_dce/"