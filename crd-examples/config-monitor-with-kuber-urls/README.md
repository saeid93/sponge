# ConfigMonitor Operator

## Problem Statement
Assume you have a `Deployment` in a Kubernetes cluster, this deployment have to read the configuration parameters from a `ConfigMap`. You deploy the application on the cluster. After a few weeks, you have to change the configuration parameters in the `ConfigMap`, but when you do so, the `Deployment` do not restart the `Pods` in order to re-read the configuration parameters from the `ConfigMap`. Therefore, you have restart the `Pods` by yourself. In this regards, we want to create an `Operator` to look for the changes in the corresponding `ConfigMap`, and restart the `Pods`.


## Project Explanation
Here, We explain all the directories and files in the project.

### app directory
```
└── app
	├── app.py
	├── configmap.yml
	├── deploy.yml
	└── Dockerfile
```
1. app.py: There is a simple API application in the `app` directory. When you call the API, it reads the `MSG` value from the a file in `/config/config.cfg` which is mounted as a `Volume` in the `Container`. The values in the `/config/config.cfg` are written by a `ConfigMap` (in this project `configmap.yml` file)

2. configmap.yml: The `configmap.yml` is a `ConfigMap` that contains a key/value, called "MSG". When application starts to execute, first it loads the valriables from `configmap.yml`


3. deploy.yml: this file is used to deploy the application in a Kubernetes cluster. It creates a pod from `app.py`. It also contains a `Volume` to mount the `ConfigMap` which is defined in `configmap.yml`.

4. Dockerfile: a Dockerfile to build the application

### operator directory
```
└── operator
	├── main.py
	├── deploy.yml
	└── Dockerfile
```

1. main.py: it contains the logic of the operator.
2. deploy.yml: it is used to deploy the application in a Kubernetes cluster. It creates a pod from the `main.py`. It ensures that the operator is always running.
3. Dockerfile: a Dockerfile to build the application

### crd directory
```
└── crd
	├── crd.yml
	└── configmonitor.yml
```

1. crd.yml: it defines a `CustomResourceDefinition`, called `ConfigMonitor` and a `Role`. A more in-depth explanation is as follows:
```bash
Lines 2 and 3: the API version that enables you to add CRDs. CRDs are added to the cluster through a resource of their own. The resource type is CustomResourceDefinitinon.

Line 7: the scope defines whether this resource is available to the entire cluster or to the namespace where it lives. If a namespace is deleted, all the CRDs that are associated with it are deleted as well.

Line 8 and 9: the group and version define how the REST endpoint would be called. In our example, this is /apis/magalix.com/v1.

Lines 10 to 13: specify how we are going to call our new resource. A resource is called in three places:

The plural name is used in the API endpoint. In our example, that’d be /apis/magalix.com/v1/configminotors.
The singular name is the one used on the CLI, for example when using kubectl subcommands. It’s also used for displaying the results.
Lines 14 to 26: the remaining lines in the file are used to validate the resources that shall be created using this CRD. It uses the OpenAPI specification version 3. Validation is used to ensure that the correct field types and values are used. For example, the container parameter of a Pod definition is expecting an array of container objects. Adding a single string value would break things so validation ensures that if an incorrect value was entered, the API would refuse the definition and the resource is not created. In our case, we are expecting:

A configmap parameter: a string that specifies the name of the ConfigMap resource.
A podSelector parameter: this is a placeholder for the label(s) that would be used to select the Pods. The label itself is a string so the object is permitted to have one or more string values by specifying the additionalProperties parameter.

The second part of the file contains a definition for the role we’re using with the CRD. The role allows the CRD access to the verbs on the API group
```
2. configmonitor.yml: this is an instance of the `ConfigMonitor` CRD.

## Run the Project in Debug Mode

1. First, create the `ConfigMonitor` crd as follows:
```bash
kubectl apply -f crd/crd.yml
```

2. create the `ConfigMonitor` resource as follows:
```bash
kubectl apply -f crd/configmonitor.yml
```

3. create the `ConfigMap` as follows:
```bash
kubectl apply -f app/configmap.yml
```

4. build the application:
```bash
docker build <docker-registry>/frontend:latest .
```

5. push the image in your docker registry:
```bash
docker push <docker-registry>/frontend:latest
```

6. run the application as follows:
```bash
kubectl apply -f app/deploy.yml
```

7. open a termial and execute the following command to access the Kubernetes API:
```bash
kubectl proxy
```

8. run the operator:
```bash
python3 operator/main.py
```

9. edit the value of `MSG` in `app/configmap.yml` file and then apply the change as follows:
```bash
kubectl apply -f app/configmap.yml
```

10. check the API to be sure of the changes:
```bash
kubectl exec -it <app-container-name> -- curl localhost
```

11. enjoy!

## Run the Project in Production

1. First, create the `ConfigMonitor` crd as follows:
```bash
kubectl apply -f crd/crd.yml
```

2. create the `ConfigMonitor` resource as follows:
```bash
kubectl apply -f crd/configmonitor.yml
```

3. create the `ConfigMap` as follows:
```bash
kubectl apply -f app/configmap.yml
```

4. build the application:
```bash
docker build <docker-registry>/frontend:latest .
```

5. push the image in your docker registry:
```bash
docker push <docker-registry>/frontend:latest
```

6. run the application as follows:
```bash
kubectl apply -f app/deploy.yml
```

7. open a termial and execute the following command to access the Kubernetes API:
```bash
kubectl proxy
```

8. build the operator:
```bash
docker build <docker-registry>/operator:latest .
```

9. push the image in our local registry:
```bash
docker push <docker-registry>/operator:latest
```

10. run the operator deployment:
```bash
kubectl apply -f operator/deploy.yml
```

11. edit the value of `MSG` in `app/configmap.yml` file and then apply the change as follows:
```bash
kubectl apply -f app/configmap.yml
```

12. check the API to be sure of the changes:
```bash
kubectl exec -it <app-container-name> -- curl localhost
```

13. enjoy!


## References

1. [Create a Custom Operator](https://www.magalix.com/blog/creating-custom-kubernetes-operators)
2. [What is an Operator](https://www.redhat.com/en/topics/containers/what-is-a-kubernetes-operator)
3. [Custom Resource Definitions](https://kubernetes.io/docs/concepts/extend-kubernetes/api-extension/custom-resources/)
4. [Operators](https://blog.container-solutions.com/kubernetes-operators-explained)
5. [CRDs](https://www.bmc.com/blogs/kubernetes-crd-custom-resource-definitions/)
6. [RBAC](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)