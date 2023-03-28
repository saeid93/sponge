#!/bin/bash

# Find all bash scripts in subfolders
scripts=$(find . -type f -name "*.sh")

# Loop through each script and run it
for script in $scripts
do
    echo "Running script: $script"
    bash $script
done
