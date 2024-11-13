#!/bin/bash

# Check that two arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <output_folder> <inference_script_dir>"
    exit 1
fi

# Assign arguments to variables
input_folder="$1"
output_folder="$2"
inference_script_dir="$3"

# Activate the conda environment
source $HOME/miniforge3/bin/activate 2dssg

# Run the inference script
cd $inference_script_dir
python inference.py --config ./configs/config_3DSSG_full_l160_pcl.yaml \
                    --loadbest 1 \
                    -o $output_folder \
                    --dry_run \
                    --data_path $input_folder \
                    --model_dir ./models/3dssg_160

