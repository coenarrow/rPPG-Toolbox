#!/bin/bash

# Check if a mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {conda}"
    exit 1
fi

MODE=$1

module load cuda/11.8

# Function to set up using conda
conda_setup() {
    echo "Setting up using conda..."
    conda init bash
    source ~/.bashrc
    conda remove --prefix ./env --all -y || exit 1
    conda create --prefix ./env python==3.10 -y
    chmod -R u+rwx ./env
    conda activate ./env
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    cd tools/mamba || exit 1
    python -m pip install . || exit 1
}

# Execute the appropriate setup based on the mode
case $MODE in
    conda)
        conda_setup
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 {conda}"
        exit 1
        ;;
esac
