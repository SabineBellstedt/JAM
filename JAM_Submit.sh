#!/bin/bash
# specify the queue name
#PBS -q gstar
# resource requests
#PBS -l walltime=00:02:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=1gb

# run process
module load python
# module load mpi4py
cd /nfs/cluster/gals/sbellstedt/Analysis/JAM/
python /nfs/cluster/gals/sbellstedt/Analysis/JAM/JAM_Sabine_General.py NGC1023 10