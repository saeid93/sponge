#!/bin/bash

output_file="all_deployments_audio.yaml"

# Get a list of all Seldon Deployments
deployments=$(kubectl get seldondeployments -o name)

# Iterate through each deployment and retrieve its YAML configuration
for deployment in $deployments; do
    deployment_config=$(kubectl get $deployment -o yaml)
    echo "$deployment_config" >> "$output_file"
done

echo "All Seldon Deployment configurations have been saved to $output_file."
