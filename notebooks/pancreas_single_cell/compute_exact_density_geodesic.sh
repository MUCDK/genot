#!/bin/bash
#SBATCH --constraint=a100_40gb|a100_80gb
#SBATCH -J exact_density_geodesic
#SBATCH -o exact_density_geodesic_out.txt
#SBATCH -e exact_density_geodesic_err.txt
#SBATCH -q gpu_long
#SBATCH -t 48:00:00
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH --nice=1

source ${HOME}/.bashrc_new
conda activate /home/icb/dominik.klein/mambaforge/envs/entot_pip

python /home/icb/dominik.klein/git_repos/entot/notebooks/pancreas_single_cell/compute_exact_density_geodesic.py

