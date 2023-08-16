#!/bin/bash

function fetch_pipeline(){
    output_file="all_deployments_audio.yaml"
    deployments=$(kubectl get seldondeployments -o name)
    for deployment in $deployments; do
        deployment_config=$(kubectl get $deployment -o yaml)
        echo "$deployment_config" >> "$output_file"
    done
    echo "All Seldon Deployment configurations have been saved to $output_file."
}

fetch_pipeline