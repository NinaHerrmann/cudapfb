#!/bin/bash
#SBATCH --job-name pfb64-GPU-nodes-1-gpu-1
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --exclusive
#SBATCH --output /home/nihe532b/musket-build/pfb64/CUDA/out/pfb64-nodes-1-gpu-1.out
#SBATCH --cpus-per-task 24
#SBATCH --mail-type ALL
#SBATCH --mail-user nihe532b@mailbox.tu-dresden.de
#SBATCH --time 00:10:00
#SBATCH -A p_telescope
#SBATCH --gres gpu:4

export OMP_NUM_THREADS=24

RUNS=10
for ((i=1;i<=RUNS;i++)); do
    srun --multi-prog /home/nihe532b/musket/src-gen/pfb64/CUDA/job.conf
done
