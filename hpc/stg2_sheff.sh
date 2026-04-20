#!/bin/bash
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=stg2
#SBATCH --output=/mnt/parscratch/users/eia21frd/ppln-ibeat-dce/hpc/logs/stg2.out
#SBATCH --error=/mnt/parscratch/users/eia21frd/ppln-ibeat-dce/hpc/logs/stg2.err
#SBATCH --mail-user=frsdavies1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END

# Environment setup
module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

# Path definitions
ENV="/mnt/parscratch/users/$(whoami)/envs/dce"
CODE="/mnt/parscratch/users/$(whoami)/ppln-ibeat-dce/src/ibeat_dce"

# 1. FIX: Navigate to the code directory (using the correct variable name)
cd "$CODE"

# 2. FIX: Run the python script using the local path now that we are in the folder
srun "$ENV/bin/python" "stage_2_shef.py"

# 3. FIX: Ensure the output directory exists before rsync runs
# (If Python finds 0 files, this folder might not exist, causing rsync to fail)
mkdir -p "/mnt/parscratch/users/$(whoami)/build/stage_2_compute_descriptivemaps/"

# Sync results
rsync -av --no-group --no-perms "/mnt/parscratch/users/$(whoami)/build/stage_2_compute_descriptivemaps/" "login1:/shared/abdominal_imaging/Shared/ibeat_dce/results/stage_2/"