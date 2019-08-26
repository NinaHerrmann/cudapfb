#!/bin/bash
#SBATCH --job-name lpfb64
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --gres gpu:4
#SBATCH --exclusive
#SBATCH --exclude taurusi[2001-2044]
#SBATCH --output l64.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user nina.herrmann@mailbox.tu-dresden.de
#SBATCH --time 00:20:00
#SBATCH -A p_telescope

RUNS=2
for ((i=1;i<=RUNS;i++)); do
    srun /home/nihe532b/l64
done
