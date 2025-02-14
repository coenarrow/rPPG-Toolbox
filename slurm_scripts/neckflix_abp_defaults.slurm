#!/bin/bash
#SBATCH --job-name=abp_defaults
#SBATCH --output=slurm_out/%A_Neckflix_abp_defaults.out
#SBATCH --error=slurm_out/%A_Neckflix_abp_defaults.err
#SBATCH --partition=pophealth
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1

REPO_DIR="/group/pgh004/carrow/repo/rPPG-Toolbox"
CONFIG_DIR="/group/pgh004/carrow/repo/rPPG-Toolbox/configs/neckflix_configs/abp/defaults"
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

CONFIGS=$(find $CONFIG_DIR -type f -name '*.yaml')

# Loop through each file in the list.
for config in $CONFIGS; do
  echo "Running config: $config"
  echo "Start Time: $(date)"
  python ${SCRIPT} --config_file "$config"
  echo "Completed: $config"
  echo "End Time: $(date)"
done