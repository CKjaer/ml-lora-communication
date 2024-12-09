#!/bin/bash
#SBATCH --job-name=Symb_Generator
#SBATCH --output=./generate_training_data/job_results/result_%j.out
#SBATCH --error=./generate_training_data/job_results/error_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

singularity shell --nv "$(pwd)/tensorflow_24.09.sif" << 'EOF'
BASE_DIR="$(pwd)"
echo "Currently in directory: $BASE_DIR - Should start from ml-lora-communication"

# Run Python file (generate_data.py)
python -u "$BASE_DIR/generate_training_data/generate_data.py" || { echo "Python script failed"; exit 1; }

# Exit the container shell

exit
EOF
