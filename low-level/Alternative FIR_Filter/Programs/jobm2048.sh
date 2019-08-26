#!/bin/bash
#SBATCH --job-name mpfb2048
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --gres gpu:4
#SBATCH --exclusive
#SBATCH --exclude taurusi[2001-2044]
#SBATCH --output m2048.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user nina.herrmann@mailbox.tu-dresden.de
#SBATCH --time 00:20:00
#SBATCH -A p_telescope

RUNS=2
for ((i=1;i<=RUNS;i++)); do
    srun /home/nihe532b/m2048
done
