#!/bin/bash
<<<<<<<< HEAD:cnn_bash/models_test.sh
#SBATCH --job-name=Test_models
#SBATCH --output=./job_results/test_model_result_%j.out
#SBATCH --error=./job_results/test_model_error_%j.err
========
#SBATCH --job-name=train_models
#SBATCH --output=./job_results/result_%j.out
#SBATCH --error=./job_results/error_%j.err
>>>>>>>> origin/main:cnn_bash/train_models.sh
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

singularity shell --nv "$(pwd)/pytorch_24.09.sif" << 'EOF'
BASE_DIR="$(pwd)"
echo "Currently in directory: $BASE_DIR"

<<<<<<<< HEAD:cnn_bash/models_test.sh
# Run Python file (model_test.py)
python -u "$BASE_DIR/cnn_bash/model_test.py" || { echo "Python script failed"; exit 1; }
========
# Run Python file (evalulate_models.py)
python -u "$BASE_DIR/cnn_bash/train_models.py" || { echo "Python script failed"; exit 1; }
>>>>>>>> origin/main:cnn_bash/train_models.sh

# Exit the container shell

exit
EOF
