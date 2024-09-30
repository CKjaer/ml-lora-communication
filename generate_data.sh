#!/bin/bash
#SBATCH --job-name=726_Symbol_Generator
#SBATCH --output=result_%j.out
#SBATCH --error=error_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

singularity shell ./tensorflow_24.07.sif << 'EOF'

# 2. Create virtual environment (if it doesn't already exist)
VENV_DIR="$HOME/virtualenv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# 3. Enter virtual environment
source "$VENV_DIR/bin/activate"

# 4. Install requirements via pip
REQUIREMENTS_FILE="$HOME/requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    pip install -r "$REQUIREMENTS_FILE" || { echo "Failed to install requirements"; exit 1; }
else
    echo "Requirements file not found: $REQUIREMENTS_FILE"
    exit 1
fi

# 5. Run Python file (main.py)
python "$HOME/main.py" || { echo "Python script failed"; exit 1; }

# 6. Find the newest folder in $HOME/output/
OUTPUT_DIR="$HOME/output"
NEWEST_FOLDER=$(ls -dt "$OUTPUT_DIR"/*/ | head -1)
echo "Newest folder: $NEWEST_FOLDER"

# 7. Tar the contents of the newest folder
tar -czvf "$OUTPUT_DIR/latest_folder.tar.gz" -C "$NEWEST_FOLDER" . || { echo "Failed to create tar file"; exit 1; }

# Deactivate the virtual environment
deactivate

# Exit the container shell
exit
EOF
