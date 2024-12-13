#!/bin/bash
#SBATCH --job-name=726SimSNR
#SBATCH --output=./simulate_monte_carlo/job_results/SNR_result_%j.out
#SBATCH --error=./simulate_monte_carlo/job_results/SNR_error_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

echo "Started from $(pwd) - should be started from ml-lora-communication folder"
singularity shell --nv "$(pwd)/tensorflow_24.09.sif" << 'EOF'


# Run Python file (main.py)
python -u "$(pwd)/simulate_monte_carlo/simulate_snr_values.py" || { echo "Python script failed"; exit 1; }

# Exit the container shell

exit
EOF