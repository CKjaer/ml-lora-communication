#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --output=./job_results/IQ_result_%j.out
#SBATCH --error=./job_results/IQ_error_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G

singularity shell --nv "$(pwd)/pytorch_24.09.sif" << 'EOF'
BASE_DIR="$(pwd)"
echo "Currently in directory: $BASE_DIR"

# Run Python file (evalulate_models.py)
python -u "$BASE_DIR/IQ_cnn/iq_cnn_train.py" || { echo "Python script failed"; exit 1; }

# Exit the container shell

exit
EOF
