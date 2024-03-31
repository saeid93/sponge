#!/bin/bash

install_network_tools() {
    sudo apt install iproute2 iperf
    sudo ufw allow 5001/udp
    sudo ufw allow 5001/tcp
    echo "End Install Network tools"
    echo
}


function install_istio() {
    echo "Install Istio"
    minikube addons enable istio-provisioner
    minikube addons enable istio
    # kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/prometheus.yaml
    # kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
    # script_dir=$(dirname "$0")
    # cd "$script_dir"
    # kubectl apply -f istio-monitoring.yaml
    echo "End Install Istio"
    echo
}


function configure_monitoring() {
    echo "Configure monitoring"

    minikube kubectl create namespace monitoring
    echo "Adding Prometheus Helm repository..."
    # helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

    # echo "Installing Prometheus..."
    # helm install prometheus prometheus-community/prometheus -n monitoring
    # helm install prometheus prometheus-community/prometheus -n monitoring
    helm upgrade --install monitoring kube-prometheus \
        --version 8.3.2 \
        --set fullnameOverride=monitoring \
        --namespace monitoring \
        --repo https://charts.bitnami.com/bitnami

    cat <<EOF | minikube kubectl apply -- -f -
apiVersion: monitoring.coreos.com/v1
kind: PodMonitor
metadata:
  name: seldon-podmonitor
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app.kubernetes.io/managed-by: seldon-core
  podMetricsEndpoints:
    - port: metrics
      interval: 1s
      path: /prometheus
  namespaceSelector:
    any: true
EOF


    minikube kubectl apply -- -f ~/infrastructure/istio-monitoring.yaml
    minikube kubectl patch -- svc monitoring-prometheus -n monitoring --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
    # minikube kubectl patch -- svc grafana -n monitoring --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
    minikube kubectl patch -- svc monitoring-prometheus -n monitoring --patch '{"spec": {"type": "NodePort", "ports": [{"port": 9090, "nodePort": 30090}]}}'
    # minikube kubectl patch -- svc grafana -n monitoring --patch '{"spec": {"type": "NodePort", "ports": [{"port": 3000, "nodePort": 30300}]}}'
    # minikube kubectl apply -- -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
    # minikube kubectl apply -- -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/jaeger.yaml
    echo "End Configure monitoring"k
    echo
}


configure_monitoring
install_istio
install_network_tools