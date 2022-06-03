#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output=extra_credit/apf.%j.%N.nprocs=1.px=1.py=1.i=100.n=800.k=0.ref=0.partition=CLUSTER.out
#SBATCH --partition=CLUSTER
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --mem=512M
# #SBATCH --account=csd720
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:00:30
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=bsaldanha@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
# module purge
# module load cpu
#Load module file(s) into the shell environment
# module load gcc
# module load rocks-openmpi
# module load slurm
# srun --mpi=pmi2 -n 1 ./apf -n 800 -i 2000
mpirun -n 1 ./apf -n 800 -i 100 -x 1 -y 1 -p 100
