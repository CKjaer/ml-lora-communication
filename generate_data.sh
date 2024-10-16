#!/bin/bash
#SBATCH --job-name=726_Symbol_Generator
#SBATCH --output=./job_results/result_%j.out
#SBATCH --error=./job_results/error_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

singularity shell --nv "$(pwd)/tensorflow_24.09.sif" << 'EOF'


# Run Python file (main.py)
python "$BASE_DIR/main.py" || { echo "Python script failed"; exit 1; }

# Exit the container shell

exit
EOF
