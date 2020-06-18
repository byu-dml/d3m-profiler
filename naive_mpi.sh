#!/bin/bash

#SBATCH --time=30:00   # walltime
#SBATCH --ntasks=50   # number of processor cores (i.e. tasks)
#SBATCH --nodes=5  # number of nodes
#SBATCH --mem-per-cpu=6500M   # memory per CPU core
#SBATCH -J "Naive Tester"   # job name

#add the right thing to the file path
export PATH="$HOME/.local/bin:$PATH"

#load the modules
module load python/3.8
module load mpi/openmpi-1.10.7_gnu4.8.5

#now run the script
mpirun python naive_mpi.py
