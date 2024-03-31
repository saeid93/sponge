#!/bin/bash

function clean_cluster() {
    local force_flag=""
    if [ "$1" = "force" ]; then
        force_flag="--force"
    fi
    
    minikube kubectl -- delete seldondeployment --all $force_flag -n default
    minikube kubectl -- delete deployments --all $force_flag -n default
    minikube kubectl -- delete replicaset --all $force_flag -n default
    minikube kubectl -- delete pods --all $force_flag -n default
    minikube kubectl -- delete services --all $force_flag -n default
    minikube kubectl -- get services | grep -v kubernetes | awk '{print $1}' | xargs minikube kubectl -- delete service -n default
}

clean_cluster
