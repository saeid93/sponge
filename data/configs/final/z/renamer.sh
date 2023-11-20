#!/bin/bash

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <old-suffix> <new-suffix>"
    exit 1
fi

old_suffix="$1"
new_suffix="$2"

# Loop through all files in the current directory ending with old_suffix.yaml
for file in *"$old_suffix.yaml"; do
    # Check if the file is not the script itself
    if [ "$file" != "rename_script.sh" ]; then
        # Replace old_suffix.yaml with new_suffix.yaml in the filename
        new_name="${file//$old_suffix.yaml/$new_suffix.yaml}"
        # Rename the file
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    fi
done

