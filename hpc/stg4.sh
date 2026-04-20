#!/bin/bash
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=stg4
#SBATCH --output=logs/stg4.out
#SBATCH --error=logs/stg4.err

module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

ENV="/mnt/parscratch/users/$(whoami)/envs/dce"
CODE="/mnt/parscratch/users/$(whoami)/ppln-ibeat-dce/src/ibeat_dce"


# Run Stage 4
srun "$ENV/bin/python" "$CODE/stage_4_Motion_Correct.py"

# Sync Final Motion Corrected results to shared drive
rsync -av --no-group --no-perms "/mnt/parscratch/users/$(whoami)/build/stage_4_motioncorrected" "login1:/shared/abdominal_imaging/Shared/ibeat_dce/results/stage_4"