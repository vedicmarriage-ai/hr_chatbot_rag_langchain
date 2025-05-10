#!/bin/bash

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm

# Execute the command passed as arguments
exec "$@" 