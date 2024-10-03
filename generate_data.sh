#!/bin/bash
#SBATCH --job-name=726_Symbol_Generator
#SBATCH --output=./job_results/result_%j.out
#SBATCH --error=./job_results/error_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

#pyhon3 -m venv virtualenv

singularity shell --nv "$(pwd)/tensorflow_24.07.sif" << 'EOF'
BASE_DIR="$(pwd)"

# 2. Create virtual environment
#VENV_DIR="$BASE_DIR/virtualenv"
#python3 -m venv "$VENV_DIR"

# 3. Enter virtual environment
#source "$VENV_DIR/bin/activate"

# 4. Install requirements via pip
#REQUIREMENTS_FILE="$BASE_DIR/requirements.txt"
#if [ -f "$REQUIREMENTS_FILE" ]; then
#    pip install -r "$REQUIREMENTS_FILE" || { echo "Failed to install requirements"; exit 1; }
#else
#    echo "Requirements file not found: $REQUIREMENTS_FILE"
#    exit 1
#fi

# 5. Run Python file (main.py)
python "$BASE_DIR/main.py" || { echo "Python script failed"; exit 1; }

# Deactivate the virtual environment
deactivate

# Exit the container shell
exit

EOF
