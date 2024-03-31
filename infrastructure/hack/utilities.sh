#!/bin/bash

install_kubectl() {
    echo "Install kubectl"
    curl -LO https://dl.k8s.io/release/v$K8SVERSION.2/bin/linux/amd64/kubectl
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    sudo microk8s config > $HOME/.kube/config
    sudo ufw allow 16443
    echo "y" | sudo ufw enable
    echo "End Install kubectl"
    rm kubectl
}

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

function install_seldon_core() {
    echo "Install Seldon Core"
    kubectl create namespace seldon-system
    helm install seldon-core seldon-core-operator \
        --repo https://storage.googleapis.com/seldon-charts \
        --set usageMetrics.enabled=true \
        --set istio.enabled=true \
        --namespace seldon-system

    cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: seldon-gateway
  namespace: istio-system
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
EOF

    kubectl patch svc istio-ingressgateway -n istio-system --patch '{"spec": {"ports": [{"name": "http2", "nodePort": 32000, "port": 80, "protocol": "TCP", "targetPort": 8080}]}}'
    echo "End Install Seldon Core"
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

    else
        echo "Enabling Prometheus addon in microk8s..."
        sudo microk8s enable prometheus
    fi


      minikube kubectl apply -- -f ~/infrastructure/istio-monitoring.yaml
      minikube kubectl patch -- svc monitoring-prometheus -n monitoring --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
      # minikube kubectl patch -- svc grafana -n monitoring --type='json' -p '[{"op":"replace","path":"/spec/type","value":"NodePort"}]'
      minikube kubectl patch -- svc monitoring-prometheus -n monitoring --patch '{"spec": {"type": "NodePort", "ports": [{"port": 9090, "nodePort": 30090}]}}'
      # minikube kubectl patch -- svc grafana -n monitoring --patch '{"spec": {"type": "NodePort", "ports": [{"port": 3000, "nodePort": 30300}]}}'
      # minikube kubectl apply -- -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/kiali.yaml
      # minikube kubectl apply -- -f https://raw.githubusercontent.com/istio/istio/release-1.13/samples/addons/jaeger.yaml
    echo "End Configure monitoring"
    echo
}


configure_monitoring
install_network_tools