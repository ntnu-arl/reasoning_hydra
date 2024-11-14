#!/bin/bash

# Default path where to put third_party libraries
DEFAULT_THIRD_PARTY_DIR="$HOME/workspaces/hierarchical_reasoning_ws/src/reasoning_hydra/third_party"

# Use the provided argument as the path, or use the default path if no argument is provided
THIRD_PARTY_DIR="${1:-$DEFAULT_THIRD_PARTY_DIR}"

# Create the path if it does not exist
if [ ! -d "$THIRD_PARTY_DIR" ]; then
    mkdir -p "$THIRD_PARTY_DIR"
fi

# Navigate to the third_party directory
cd "$THIRD_PARTY_DIR"

# Install DBoW2
if [ ! -d "DBoW2" ]; then
    git clone https://github.com/dorian3d/DBoW2.git
    cd DBoW2
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    cd ../..
fi

# Install opengv
if [ ! -d "opengv" ]; then
    git clone https://github.com/laurentkneip/opengv.git
    cd opengv
    mkdir build
    cd build
    cmake ..
    make
    sudo make install
    cd ../..
fi

echo "Third-party dependencies installed successfully."
