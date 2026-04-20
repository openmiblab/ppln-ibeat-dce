#!/bin/bash   
#SBATCH --mem=32G         
#SBATCH --cpus-per-task=1
#SBATCH --time=36:00:00
#SBATCH --mail-user=frsdavies1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=patch_times
#SBATCH --output=logs/patch_times.out
#SBATCH --error=logs/patch_times.err

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can 
# prevent performance issues if your code manages threading itself 
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND

# Ensures that all your environment variables from the submission 
# environment are passed into the job’s environment
export SLURM_EXPORT_ENV=ALL

# Loads the Anaconda module provided by the cluster.
# (On HPC systems, software is usually installed as “modules” to avoid version conflicts.)
module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC

# Tell VTK/PyVista to use the OSMesa (Off-Screen) library
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

# Define path variables here
ENV="/mnt/parscratch/users/$(whoami)/envs/dce"
CODE="/mnt/parscratch/users/$(whoami)/ppln-ibeat-dce/src/ibeat_dce"

# srun runs your program on the allocated compute resources managed by Slurm
srun "$ENV/bin/python" "$CODE/patch2.py"
rsync -av --no-group --no-perms "/mnt/parscratch/users/$(whoami)/build/stage_2_compute_descriptivemaps/" "login1:/shared/abdominal_imaging/Shared/ibeat_dce/results/stage_2/"