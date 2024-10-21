#!/bin/bash
#SBATCH --constraint=a100_40gb|a100_80gb
#SBATCH -J test_b
#SBATCH -o test_b_out.txt
#SBATCH -e test_b_err.txt
#SBATCH -q gpu_normal
#SBATCH -t 08:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160GB
#SBATCH --nice=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/icb/dominik.klein/mambaforge/envs/entot_pip_copy

python /home/icb/dominik.klein/git_repos/genot_benchmarks/neurips_rebuttal/test_benchmark.py 

     