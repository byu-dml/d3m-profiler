#!/bin/bash

#SBATCH --time=12:00:00   # walltime
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=12288M   # memory per CPU core
#SBATCH -J "Profiler Tester"   # job name

#add the right thing to the file path
export PATH="$HOME/.local/bin:$PATH"

#load the modules
module load python/3.8

#now run the script
python -u example.py
