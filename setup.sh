#!/bin/bash

# 1. Activate the virtual environment
source venv/bin/activate

# 2. Set PYTHONPATH to your project root
# (so Python knows to find modules like "data" and "src")
export PYTHONPATH=$(pwd)

# 3. Optional: Start Jupyter Lab
# jupyter lab

echo "✅ Virtual environment activated and PYTHONPATH set!"
echo "Current PYTHONPATH: $PYTHONPATH"
