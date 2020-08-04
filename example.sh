#!/bin/bash

#SBATCH --time=14:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=2  # number of nodes
#SBATCH --gres=gpu:2 #number of gpus
#SBATCH --mem-per-cpu=10000M   # memory per CPU core
#SBATCH -J "Total Tester"   # job name

#add the right thing to the file path
export PATH="$HOME/.local/bin:$PATH"

#load the modules
module load  miniconda3/4.6 
#activate the environment
source activate cross
#load the remaining modules
module load mpi/openmpi-1.10.7_gnu4.8.5
module load compiler_gnu/6.4
module load cuda/10.1
module load cudnn/7.6

#now run the script
mpirun python -u example.py
