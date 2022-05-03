### An example model deployment using torch and seldon-core

### prerequisites:
- docker
- a kubernetes cluster >= 1.18
- helm >= 3.0
### steps:
-   ```shell
    kubectl create namespace seldon-system
    ```
-   ```shell
    helm install seldon-core seldon-core-operator \
    --repo https://storage.googleapis.com/seldon-charts \
    --set usageMetrics.enabled=true \
    --set istio.enabled=true \
    --namespace seldon-system
    ```
- ```shell
    docker build . -t torch_resnet:v1
    ```
- ```shell 
  kubectl apply -f deploy.yml
  ```
- expose the deployment. example: 
    ```shell
    kubectl expose deployment DEPLOYMENT_NAME --type NodePort --port 9000 --target-port 9000 --name torch-resnet-svc
    ```
- get nodePort:
    ```shell
    kubectl get svc torch-resnet-svc -o json | jq '.spec.ports[0].nodePort'
    ```
- test the deployment:
    ```shell
    curl <worker_node_ip>:<port_output_by_above_command>/health/status
    ```