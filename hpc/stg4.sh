#!/bin/bash
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=stg4
#SBATCH --output=logs/stg4.out
#SBATCH --error=logs/stg4.err

module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

ENV="/mnt/parscratch/users/$(whoami)/envs/dce"
CODE="/mnt/parscratch/users/$(whoami)/ppln-ibeat-dce/src/ibeat_dce"

# Sync AIF from shared drive (Stage 3) back to parscratch before starting
rsync -av --no-group --no-perms "login1:/shared/abdominal_imaging/Shared/ibeat_dce/data/ibeat_dce/stage_2_descriptive/" "/mnt/parscratch/users/$(whoami)/data/ibeat_dce/stage_2_descriptive/"

# Run Stage 4
srun "$ENV/bin/python" "$CODE/stage_4_moco.py"

# Sync Final Motion Corrected results to shared drive
rsync -av --no-group --no-perms "/mnt/parscratch/users/$(whoami)/data/ibeat_dce/stage_4_moco" "login1:/shared/abdominal_imaging/Shared/ibeat_dce/data/ibeat_dce/"