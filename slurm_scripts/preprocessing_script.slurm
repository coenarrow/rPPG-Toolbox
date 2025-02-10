#!/bin/bash
#SBATCH --job-name=Preprocessing_Script
#SBATCH --output=slurm_out/%A_Preprocessing.out
#SBATCH --error=slurm_out/%A_Preprocessing.err
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=8
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

echo "Current date: $(date)"

echo "Running with DeepPhys"
python ${SCRIPT} --config_file "./configs/neckflix_configs/Neckflix_train_test-DeepPhys.yaml"
echo "DeepPhys Complete"

# echo "Running with EfficientPhys"
# python ${SCRIPT} --config_file "./configs/neckflix_configs/Neckflix_train_test-EfficientPhys.yaml"
# echo "EfficientPhys Complete"

# echo "Running with iBVPNet"
# python ${SCRIPT} --config_file "./configs/neckflix_configs/Neckflix_train_test-iBVPNet.yaml"
# echo "EfficientPhys Complete"

# echo "Running with PhysFormer"
# python ${SCRIPT} --config_file "./configs/neckflix_configs/Neckflix_train_test-PhysFormer.yaml"
# echo "PhysFormer Complete"

# echo "Running with PhysMamba"
# python ${SCRIPT} --config_file "./configs/neckflix_configs/Neckflix_train_test-PhysMamba.yaml"
# echo "PhysMamba Complete"

# echo "Running with PhysNet"
# python ${SCRIPT} --config_file "./configs/neckflix_configs/Neckflix_train_test-PhysNet.yaml"
# echo "PhysNet Complete"

# echo "Running with RhythmFormer"
# python ${SCRIPT} --config_file "./configs/neckflix_configs/Neckflix_train_test-RhythmFormer.yaml"
# echo "RhythmFormer Complete"

# echo "Running with TSCAN"
# python ${SCRIPT} --config_file "./configs/neckflix_configs/Neckflix_train_test-TSCAN.yaml"
# echo "TSCAN Complete"