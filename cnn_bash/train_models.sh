#!/bin/bash
#SBATCH --job-name=train_exp
#SBATCH --output=./job_results/train_result_%j.out
#SBATCH --error=./job_results/train_error_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

singularity shell --nv "$(pwd)/pytorch_24.09.sif" << 'EOF'
BASE_DIR="$(pwd)"
echo "Currently in directory: $BASE_DIR"

# Run Python file (evalulate_models.py)
python -u "$BASE_DIR/cnn_bash/train_models.py" || { echo "Python script failed"; exit 1; }

# Exit the container shell

exit
EOF
