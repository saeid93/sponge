#!/bin/bash

input_file="all_deployments_audio.yaml"
output_file="deployment_without_status.yaml"

# Remove 'status' field using sed
sed '/status:/,/^  state:/d' "$input_file" > "$output_file"

echo "Deployment configuration without 'status' field has been saved to $output_file."
