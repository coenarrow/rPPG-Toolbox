#!/bin/bash
#SBATCH --job-name=PURE_UBFC_EFFICIENTPHYS
#SBATCH --output=slurm_out/%A_PURE_UBFC_EFFICIENTPHYS.out
#SBATCH --error=slurm_out/%A_PURE_UBFC_EFFICIENTPHYS.err
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1

REPO_DIR="/group/pgh004/carrow/repo/rPPG-Toolbox"
export PYTHONPATH=$PYTHONPATH:$REPO_DIR
cd $REPO_DIR
echo "Current Working Directory: $(pwd)"

######################
### Define Variables here ###
SCRIPT="./main.py"
######################

module load Anaconda3/2024.06
module load cuda/11.8
module list

# Detect the operating system
OS=$(uname)

# Source conda for Linux/macOS or activate conda for Windows
if [[ "$OS" == "Linux" || "$OS" == "Darwin" ]]; then
    # For Linux and macOS
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ./env
else
    # For Windows
    eval "$(conda shell.bash hook)"
    conda activate ./env
fi

echo "Activated Conda Environment"
conda info --envs

echo "Running with EfficientPhys"
python ${SCRIPT} --config_file "./configs/my_train_configs/PURE_PURE_UBFC-rPPG_EFFICIENTPHYS.yaml"
echo "EfficientPhys Complete"