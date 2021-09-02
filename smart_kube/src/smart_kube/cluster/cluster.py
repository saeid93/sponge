"""main cluster capabilities of the simulator
"""
from tempfile import TemporaryFile
import tarfile
import time

from kubernetes.client.rest import ApiException
from kubernetes.client import (
    V1ResourceRequirements,
    CustomObjectsApi,
    V1ObjectMeta,
    V1Container,
    V1Namespace,
    V1PodSpec,
    CoreV1Api,
    AppsV1Api,
    V1ContainerPort,
    V1PodTemplateSpec,
    V1Deployment,
    V1DeleteOptions,
    V1DeploymentSpec,
    V1Pod,
    V1Service,
    V1ServiceSpec,
    V1ServicePort
)
from kubernetes import config, stream
from typing import Dict

from smart_kube.util import logger


# TODO add making the cluster from the code rather than the console here

def get_service_name(service: V1Service) -> str:
    """Get name of a Service

    :param service: V1Service
        service object

    :return str
    """
    return service.metadata.name


class Cluster:

    UTILIZATION_NODE_PORT = 30000

    DATA_DESTINATION = "/"
    WORKLOAD_NAME = 'workloads.pickle'

    def __init__(
        self,
        namespace: str = "vpa-experiment",
        config_file_path: str = "~/.kube/config",
    ):
        """cluster base functionalities

        Args:
            API (CoreV1Api): type of the kubernetes python api
            ObjectAPI (CustomObjectsApi): type of kuberentes object api
            namespace (str): namespace of the kuberenes
            config_file_path (str): path to the config file
        """
        logger.info("creating cluster")
        self.config_file_path = config_file_path
        config.load_kube_config(self.config_file_path)
        # kubernetes general API
        self._core_api: CoreV1Api = CoreV1Api()
        # kuberente object api for metrics
        self._objects_api: CustomObjectsApi = CustomObjectsApi()
        # kuberentes api for creating deployments
        self._apps_api: AppsV1Api = AppsV1Api()
        self.namespace: str = namespace
        self._create_namespace(self.namespace)
        logger.info("cluster set up successfully")

    # --------------- cluster operations ---------------

    def _create_namespace(self, namespace: str):
        """Check namespace

        if namespace does not exist, it will be create.

        Args:
            namespace (str): name of the namespace
        """
        try:
            ns = self._core_api.read_namespace(namespace)
            logger.info("namespace {} already exist in the cluster")
            return ns
        except ApiException:
            logger.warn(
                "namespace <{}> does not exist, so it will be created.".format(
                    namespace
                )
            )

        logger.info("Creating Namespace <{}>".format(namespace))

        try:
            self._core_api.create_namespace(
                V1Namespace(metadata=V1ObjectMeta(name=namespace))
            )
        except ApiException as e:
            logger.error(e)

        logger.info("Waiting for creating namespace <{}>".format(namespace))

        while True:
            time.sleep(1)
            try:
                ns = self._core_api.read_namespace(namespace)
                if ns.status.phase == "Active":
                    logger.info("namespace <{}> created successfully".format(
                        namespace))
                    return ns
            except ApiException:
                logger.warn(
                    "namespace <{}> not exist, it will be created.".format(
                        namespace)
                )

    def clean(self, namespace: str = None, clean_namespace: bool = True):
        """Clean all Pods of a namespace

        Args:
            namespace (str): name of the namespace
            clean_namespace (bool): clean the namespace
        """
        if namespace is None:
            namespace = self.namespace

        # delete all deployments
        logger.info("Terminating Deployments...")
        self._apps_api.delete_collection_namespaced_deployment(namespace)

        # delete all pods
        logger.info("Terminating Pods...")
        self._core_api.delete_collection_namespaced_pod(namespace)

        if clean_namespace:
            # remove the namespace
            logger.info("Looking for namespace <{}>".format(namespace))
            try:
                self._core_api.read_namespace(namespace)
            except ApiException:
                logger.warn("Namespace <{}> does not exist".format(namespace))
                return True
            logger.info("Removing namespace <{}>".format(namespace))
            if namespace != "default":
                self._core_api.delete_namespace(namespace)
                while True:
                    time.sleep(1)
                    try:
                        self._core_api.read_namespace(namespace)
                    except ApiException:
                        logger.warn("namespace <{}> removed.".format(
                            namespace))
                        return True
            else:
                logger.info("<default> namespace cannot be removed")

    # ------------ experimental pods and deployments creations ------------

    def create_experiment_deployment(
        self,
        deployment_name: str,
        replicas: int,
        request_mem: int,
        request_cpu: float,
        limit_mem=int,
        limit_cpu=float,
    ):
        """makes the nginx experimental pod

        Args:
            name (str): name of the deployment
            replicas (int): number of pod replication
            request_mem (float): requested memeory by the
            experimental deployment
            request_cpu (float): requested cpu by the experimental deployment
            limit_mem (float): limit of the memory
            limit_cpu (float): limit of the cpu
        """

        self.create_deployment(
            deployment_name=deployment_name,
            image="nginx",
            replicas=replicas,
            deployment_selector={"app": deployment_name},
            pod_labels={"app": deployment_name},
            containers_name=deployment_name,
            request_mem="{}Mi".format(request_mem),
            request_cpu=str(request_cpu),
            limit_mem="{}Mi".format(limit_mem),
            limit_cpu=str(limit_cpu),
        )

    def create_experiment_pod(
        self,
        name: str,
        request_mem: float,
        limit_mem: float,
        request_cpu: float,
        limit_cpu: float,
    ):
        """makes the nginx experimental pod

        Args:
            request_mem (float): requested memeory
            request_cpu (float): requested cpu
            limit_mem (float): limit of the memory
            limit_cpu (float): limit of the cpu
        """
        self.create_pod(
            name=name,
            image="nginx",
            namespace=self.namespace,
            request_mem="{}Mi".format(request_mem),
            request_cpu=str(request_cpu),
            limit_mem="{}Mi".format(limit_mem),
            limit_cpu=str(limit_cpu),
        )

    def update_experimental_deployment_resources(
        self, cpu_request, cpu_limit, memory_request, memory_limit
    ):
        """updates the single container inside the experiment deployment
        requests and limits
        """
        raise NotImplementedError

    # --------------- deployment operations ---------------

    def create_deployment(
        self,
        deployment_name: str,
        image: str,
        namespace: str = None,
        replicas: int = None,
        deployment_selector: dict = None,
        pod_labels: dict = None,
        containers_name: str = None,
        request_mem: str = None,
        request_cpu: str = None,
        limit_mem: str = None,
        limit_cpu: str = None,
    ) -> None:
        """make a deployment

        Args:
            name (str): name of the deployment
            image (str): name of the used image
            namespace (str, optional): name of the kuberentes namespace
            replicas (int, optional): number of deployment's pod replicase
            selector (dict, optional): kubernetes selector
            request_mem (float): requested memeory
            request_cpu (float): requested cpu
            limit_mem (float): limit of the memory
            limit_cpu (float): limit of the cpu.

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """

        if replicas is None:
            replicas = 1

        if pod_labels is None:
            pod_labels = deployment_selector

        if containers_name is None:
            containers_name = deployment_name

        if namespace is None:
            namespace = self.namespace

        # limits, requests and environment variables
        limits, requests = dict(), dict()

        # assign limit memory
        if limit_mem is not None:
            limits.update(memory=limit_mem)

        # assign limit cpu
        if limit_cpu is not None:
            limits.update(cpu=limit_cpu)

        # assign requeste memory
        if request_mem is not None:
            requests.update(memory=request_mem)

        # assign request cpu
        if request_cpu is not None:
            requests.update(cpu=request_cpu)

        # Create the specification of container
        container = V1Container(
            name=containers_name,
            image=image,
            ports=[V1ContainerPort(container_port=80)],
            resources=V1ResourceRequirements(
                requests=requests,
                limits=limits,
            ),
        )

        # Create and configurate a spec section
        template = V1PodTemplateSpec(
            metadata=V1ObjectMeta(labels=pod_labels),
            spec=V1PodSpec(containers=[container]),
        )

        # Create the specification of deployment
        spec = V1DeploymentSpec(
            replicas=replicas,
            template=template,
            selector={"matchLabels": deployment_selector},
        )

        # Instantiate the deployment object
        deployment = V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=V1ObjectMeta(name=deployment_name, namespace=namespace),
            spec=spec,
        )

        try:
            new_deployment_name = deployment.metadata.name
            if new_deployment_name in self.existing_deployments:
                raise Exception(
                    f"A deployment with name <{new_deployment_name}>"
                    f" is already on the namespace <{namespace}>"
                    f", deployments in the ns:\n{self.existing_deployments}"
                    "\nTry a new name"
                )
            # Create deployement
            _ = self._apps_api.create_namespaced_deployment(
                body=deployment, namespace=namespace
            )

            logger.info(
                'Waiting for Deployment "{}" to run ...'.format(
                    deployment_name)
            )

            while True:
                time.sleep(1)
                deployment = self._apps_api.read_namespaced_deployment(
                    deployment.metadata.name, namespace
                )
                deployment_condition = deployment.status.conditions[0]
                if deployment_condition.status == "True":
                    logger.info('Deployment "{}" is Running'.format(
                        deployment_name))
                    return deployment
        except ApiException as e:
            logger.error(e)
        return None

    def update_deployment(
        self,
        deployment_name: str,
        image: str = None,
        namespace: str = None,
        replicas: int = None,
        deployment_selector: dict = None,
        pod_labels: dict = None,
        containers_name: str = None,
        request_mem: str = None,
        request_cpu: str = None,
        limit_mem: str = None,
        limit_cpu: str = None,
    ) -> None:
        """make a deployment with a single container inside it

        Args:
            name (str): name of the deployment
            image (str): name of the used image
            namespace (str, optional): name of the kuberentes namespace
            replicas (int, optional): number of deployment's pod replicase
            selector (dict, optional): kubernetes selector
            request_mem (float): requested memeory
            request_cpu (float): requested cpu
            limit_mem (float): limit of the memory
            limit_cpu (float): limit of the cpu.

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        deployment_obj = self.get_deployment_object(deployment_name)

        if deployment_selector is None:
            deployment_selector = deployment_obj.spec.selector.match_labels

        if replicas is None:
            replicas = deployment_obj.spec.replicas

        if pod_labels is None:
            pod_labels = deployment_obj.spec.selector.match_labels

        if containers_name is None:
            containers_name = deployment_name

        if namespace is None:
            namespace = self.namespace

        # limits, requests and environment variables
        resources = deployment_obj.spec.template.spec.containers[
            0].resources.to_dict()
        if "limits" in resources.keys():
            limits = resources["limits"]
        else:
            limits = dict()
        if "requests" in resources.keys():
            requests = resources["requests"]
        else:
            requests = dict()

        # assign limit memory
        if limit_mem is not None:
            limits.update(memory=limit_mem)

        # assign limit cpu
        if limit_cpu is not None:
            limits.update(cpu=limit_cpu)

        # assign requeste memory
        if request_mem is not None:
            requests.update(memory=request_mem)

        # assign request cpu
        if request_cpu is not None:
            requests.update(cpu=request_cpu)

        # Create the specification of container
        container = V1Container(
            name=containers_name,
            image=image,
            ports=[V1ContainerPort(container_port=80)],
            resources=V1ResourceRequirements(
                requests=requests,
                limits=limits,
            ),
        )

        # Create and configurate a spec section
        template = V1PodTemplateSpec(
            metadata=V1ObjectMeta(labels=pod_labels),
            spec=V1PodSpec(containers=[container]),
        )

        # Create the specification of deployment
        spec = V1DeploymentSpec(
            replicas=replicas,
            template=template,
            selector={"matchLabels": deployment_selector},
        )

        # Instantiate the deployment object
        deployment = V1Deployment(
            api_version="apps/v1",
            kind="Deployment",
            metadata=V1ObjectMeta(name=deployment_name, namespace=namespace),
            spec=spec,
        )

        try:
            new_deployment_name = deployment.metadata.name
            if new_deployment_name not in self.existing_deployments:
                raise Exception(
                    f"A deployment with name <{new_deployment_name}>"
                    f" does not exist in the namespace <{namespace}>"
                    f", Try creating it rather than updating it"
                )
            # Create deployement
            _ = self._apps_api.patch_namespaced_deployment(
                name=deployment_name, body=deployment, namespace=namespace
            )

            logger.info(
                'Waiting for Deployment "{}" to run ...'.format(
                    deployment_name)
            )

            # TODO something based-on the rolling update
            while True:
                time.sleep(1)
                deployment = self._apps_api.read_namespaced_deployment(
                    deployment.metadata.name, namespace
                )
                deployment_condition = deployment.status.conditions[0]
                if deployment_condition.status == "True":
                    logger.info(
                        'Deployment "{}" updated successfully'.format(
                            deployment_name)
                    )
                    return deployment
        except ApiException as e:
            logger.error(e)
        return None

    def delete_deployment(self, name):
        """delete a pod within a namespace

        Args:
            name (str): name of the pod
        """
        if name not in self.existing_deployments:
            raise Exception(
                f" Deployment <{name}> not in the <{self.namespace}>"
                f" is already on the namespace <{self.namespace}>"
                f", Deployments in the ns: {self.existing_deployments}"
            )
        try:
            self._apps_api.delete_namespaced_deployment(
                name,
                self.namespace,
                body=V1DeleteOptions(
                    propagation_policy="Foreground", grace_period_seconds=5
                ),
            )
        except ApiException as e:
            logger.error(e)
        try:
            while True:
                self._apps_api.read_namespaced_deployment(name, self.namespace)
                time.sleep(1)
        except ApiException:
            logger.info('Deployment "{}" deleted.'.format(name))

    # --------------- pod operations ---------------

    def create_pod(
        self,
        name: str,
        image: str,
        labels: dict = None,
        namespace: str = None,
        request_mem: str = None,
        request_cpu: str = None,
        limit_mem: str = None,
        limit_cpu: str = None,
    ) -> None:
        """creates ready to deploy pods blueprint in the kubernetes

        Args:
            name (str): name of pod
            image (str): name of image (e.g. nginx)
            labels (dict, optional): dict (default: {}) label of Pod.
            Defaults to None.
            namespace (str, optional): namespace of Pod. Defaults to None.
            request_mem (str, optional): requested memory. Defaults to None.
            request_cpu (str, optional): requested cpu. Defaults to None.
            limit_mem (str, optional): limited memory. Defaults to None.
            limit_cpu (str, optional): limited cpu. Defaults to None.

        Raises:
            Exception: [description]

        Returns:
            [type]: [description]
        """
        if labels is None:
            # set default value for labels
            labels = dict(svc="vpa")

        if namespace is None:
            namespace = self.namespace

        # limits, requests and environment variables
        limits, requests = dict(), dict()

        # assign limit memory
        if limit_mem is not None:
            limits.update(memory=limit_mem)

        # assign limit cpu
        if limit_cpu is not None:
            limits.update(cpu=limit_cpu)

        # assign requeste memory
        if request_mem is not None:
            requests.update(memory=request_mem)

        # assign request cpu
        if request_cpu is not None:
            requests.update(cpu=request_cpu)

        # create the containers
        containers = [
            V1Container(
                name=name,
                image=image,
                image_pull_policy="IfNotPresent",
                resources=V1ResourceRequirements(
                    limits=limits, requests=requests),
            )
        ]

        # create the pod spec
        pod_spec = V1PodSpec(hostname=name, containers=containers)

        # create the pod object
        pod = V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=V1ObjectMeta(
                name=name, labels=labels, namespace=namespace),
            spec=pod_spec,
        )
        return self._deploy_pod(pod)

    def _deploy_pod(self, pod: V1Pod):
        # check if a pod with the same name is not
        # in the namespace already
        try:
            new_pod_name = pod.metadata.name
            if new_pod_name in self.existing_pods:
                raise Exception(
                    f"a pod with name <{new_pod_name}>"
                    f" is already on the namespace <{self.namespace}>"
                    f", existing pods in the namespace:\n{self.existing_pods}"
                    "\nTry some name that is not already in the existing pods"
                )

            self._core_api.create_namespaced_pod(self.namespace, pod)

            logger.info('Waiting for Pod "{}" to run ...'.format(
                pod.metadata.name))

            while True:
                time.sleep(1)
                pod = self._core_api.read_namespaced_pod(
                    pod.metadata.name, self.namespace)
                if pod.status.phase == "Running":
                    logger.info('Pod "{}" is Running'.format(
                        pod.metadata.name))
                    return pod
                elif pod.status.phase == "Failed":
                    raise Exception(
                        "Pod placement failed, {}".format(pod.status.message)
                    )
                elif (
                    type(
                        pod.status.conditions[0].reason) == str and
                        pod.status.conditions[0].reason == "Unschedulable"
                ):
                    raise Exception(
                        "Pod is unschdulable {}".format(
                            pod.status.conditions[0].message
                        )
                    )
        except ApiException as e:
            logger.error(e)
            return None

    def delete_pod(self, name: str):
        """delete a pod within a namespace

        Args:
            name (str): name of the pod
        """
        if name not in self.existing_pods:
            raise Exception(
                f" pod <{name}> not in the ns <{self.namespace}>"
                f" is already on the namespace <{self.namespace}>"
                f", existing pods in the namespace: {self.existing_pods}"
            )
        try:
            self._core_api.delete_namespaced_pod(name, self.namespace)
        except ApiException as e:
            logger.error(e)
        try:
            while True:
                self._core_api.read_namespaced_pod(name, self.namespace)
                time.sleep(1)
        except ApiException:
            logger.info('Pod "{}" deleted.'.format(name))

    def update_pod(self):
        """Implement if needed"""
        # TODO implement
        raise NotImplementedError

    # --------------- create servies ---------------

    def create_service(
            self,
            name: str,
            namespace: str = None,
            portName: str = None,
            port: int = None,
            targetPort: int = None,
            portProtocol: str = None
    ):

        if namespace is None:
            # set default value for namespace
            namespace = self.namespace

        if portName is None:
            portName = 'web'

        if port is None:
            port = 80

        if targetPort is None:
            targetPort = 80

        if portProtocol is None:
            portProtocol = 'TCP'

        # create a service
        service = V1Service(
            api_version="v1",
            kind="Service",
            metadata=V1ObjectMeta(
                name=name,
                namespace=namespace
            ),
            spec=V1ServiceSpec(
                ports=[
                    V1ServicePort(
                        name=portName, protocol=portProtocol, port=port,
                        target_port=targetPort
                    )
                ],
                selector=dict(
                    svc=name
                )
            )
        )

        return self._deploy_service(service)

    def _deploy_service(self, service: V1Service, namespace: str = None):
        """Create a Service

        :param service: V1Service
            pod object

        :param namespace: str
            namespace of service
        """

        if namespace is None:
            # set default value for namespace
            namespace = self.namespace

        try:
            logger.info('Waiting for Service "{}" to run ...'.format(
                get_service_name(service)
            ))
            return self._core_api.create_namespaced_service(namespace, service)
        except ApiException as e:
            logger.error(e)
        return None

    # --------------- utilization server code ---------------

    def setup_utilization_server(self, image: str, workloads_path):
        """Setup Utilization Server

        :param image: str
            using this image to start the utilization server
        """

        # firstly clean the cluster
        # self.clean(self.namespace)

        name = "utilization-server"

        # create a container
        pod = V1Pod(
            api_version='v1',
            kind='Pod',
            metadata=V1ObjectMeta(
                name=name,
                labels=dict(
                    env='Park',
                    type=name
                ),
                namespace=self.namespace
            ),
            spec=V1PodSpec(
                hostname=name,
                containers=[
                    V1Container(
                        name=name,
                        image=image,
                        # image_pull_policy='IfNotPresent'
                    )
                ]
            )
        )

        logger.info("Creating pod '{}' ...".format(name))

        if self._deploy_pod(pod) is None:
            logger.error(
                "can't create pod '{}', so we will exit.. ".format(name))
            self.clean(self.namespace)
            exit(1)

        # create a service
        service = V1Service(
            api_version="v1",
            kind="Service",
            metadata=V1ObjectMeta(
                name=name,
                labels=dict(
                    env='Park',
                    type=name
                ),
                namespace=self.namespace
            ),
            spec=V1ServiceSpec(
                ports=[
                    V1ServicePort(
                        name="web", protocol="TCP", port=80, target_port=80,
                        node_port=self.UTILIZATION_NODE_PORT
                    )
                ],
                type='NodePort',
                selector=dict(
                    type=name
                )
            )
        )
        logger.info("Creating service '{}' ... ".format(name))
        if self._deploy_service(service) is None:
            logger.error(
                "can't create service '{}', so we will exit.. ".format(name))
            self.clean(self.namespace)
            exit(1)

        # upload workload file into container
        self.copy_file_inside_pod(
            pod_name=name,
            arcname=self.WORKLOAD_NAME,
            src_path=workloads_path,
            dest_path=self.DATA_DESTINATION,
            namespace=self.namespace
        )

    # --------------- resource usage operations ---------------

    def get_pods_metrics(self):
        """get all pods metrics"""

        def containers(metrics):
            items = metrics.get("items")
            conts = {
                item.get("metadata").get("name"): item.get(
                    "containers")[0].get("usage")
                for item in items
                if len(item.get("containers")) > 0
            }
            return conts

        try:
            while True:
                metrics = self._objects_api.list_namespaced_custom_object(
                    "metrics.k8s.io", "v1beta1", self.namespace, "pods"
                )

                if len(metrics.get("items")) > 0:
                    return containers(metrics)

                time.sleep(1)

        except ApiException as e:
            logger.error(e)
        return None

    def get_pod_metrics(self, pod_name: str):
        """gets a single pod metircs

        Args:
            pod_name (str): name of the pod
        """
        if pod_name not in self.existing_pods:
            raise ValueError(
                "pod <{}> does not exist, pods in the cluster:\n{}".format(
                    pod_name, self.existing_pods
                )
            )
        try:
            return (
                self._objects_api.get_namespaced_custom_object(
                    "metrics.k8s.io", "v1beta1", self.namespace, "pods",
                    pod_name
                )
                .get("containers")
                .pop()
                .get("usage")
            )
        except ApiException as e:
            logger.error(e)
        return None

    def get_deployments_metrics(self):
        """Implement if needed"""
        # TODO implement
        raise NotImplementedError

    def get_deployment_metrics(self):
        """Implement if needed"""
        # TODO implement
        raise NotImplementedError

    # --------------- vertical pod autoscaler operations ---------------

    def activate_builtin_vpa(
            self, deployment_name: str, update_mode: str = "Off"):
        # cpu_max: int, cpu_min: int, memory_max: int,
        # memory_min: int) -> None:
        """interface for the builtin autoscaler

        Args:
            controller_name (str): name of the deployment that we want to do
            the autoscaling on it
            update_policy (str): vertical pod autoscaler update policy
            options [Off, Initial, Auto]

        format of the generated vpa name:
            {deployment_name}-vpa

        """
        # check if the deployment exists
        if deployment_name not in self.existing_deployments:
            raise Exception(
                f" Deployment <{deployment_name}> not in <{self.namespace}>"
                f" is already on the namespace <{self.namespace}>"
                f", existing deployments: {self.existing_deployments}"
            )
        # check if the argumets are valid
        assert update_mode in [
            "Off",
            "Initial",
            "Auto",
        ], "Invalid update mode {}".format(update_mode)
        if deployment_name in self.existing_vpas.keys():
            raise ValueError(
                "Autoscaler <{}> already binded with deployment <{}>".format(
                    self.existing_vpas[deployment_name], deployment_name
                )
            )
        deployment_vpa_name = f"{deployment_name}-vpa"
        # TODO complete variables with checks
        logger.info(
            "adding builtin vpa for deployment <{}> with name <{}>".format(
                deployment_name, deployment_vpa_name
            )
        )
        manifest = {
            "apiVersion": "autoscaling.k8s.io/v1",
            "kind": "VerticalPodAutoscaler",
            "metadata": {"name": deployment_vpa_name},
            "spec": {
                "targetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": deployment_name,
                },
                "updatePolicy": {"updateMode": update_mode},
            },
        }  # TODO make it try-excpet and check if it has really been built
        self._objects_api.create_namespaced_custom_object(
            group="autoscaling.k8s.io",
            version="v1",
            namespace=self.namespace,
            plural="verticalpodautoscalers",
            body=manifest,
        )

    def delete_builtin_vpa(self, deployment_name):
        """delete a vpa assigend a deployment
        Args:
            deployment_name (str): name of the deployment that we want
            to delete it assiend vpa
        """
        if deployment_name not in self.existing_vpas.keys():
            raise ValueError(
                "no autoscaler is assigned to deployment {}".format(
                    deployment_name)
            )
        logger.info(
            "Deleting vpa <{}> assingned to deployment <{}>".format(
                self.existing_vpas[deployment_name], deployment_name
            )
        )
        self._objects_api.delete_namespaced_custom_object(
            group="autoscaling.k8s.io",
            name=self.existing_vpas[deployment_name],
            version="v1",
            namespace=self.namespace,
            plural="verticalpodautoscalers",
        )  # TODO make it try-excpet and check if it has really been built
        logger.info("vpa successfuly deleted")

    def update_builtin_vpa(self, deployment_name):
        """Implement if needed"""
        # TODO implement
        raise NotImplementedError

    def get_builin_vpa_recommendation_experimental_deployment(
        self, deployment_name
    ) -> Dict[str, Dict[str, int]]:
        """returns the upper, target and lower bound of the experimental
        deployment
        """
        response = self.get_builin_vpa_recommendation(deployment_name)
        if response is not None:
            return response[0]
        else:
            logger.info("status not yet available")
            return None

    # TODO add class resource decorators
    def get_builin_vpa_recommendation(
        self, deployment_name
    ) -> Dict[str, Dict[str, int]]:
        """returns the upper, target and lower bound of the default autoscaler
        for both cpu and memory resources
        """
        # TODO check if recommendations exists
        response = self._objects_api.get_namespaced_custom_object(
            group="autoscaling.k8s.io",
            version="v1",
            name=f"{deployment_name}-vpa",
            namespace=self.namespace,
            plural="verticalpodautoscalers",
        )
        if (
            "status" in response.keys() and "containerRecommendations"
            in response["status"]["recommendation"].keys()
        ):
            return response["status"]["recommendation"][
                "containerRecommendations"]
        else:
            logger.info("status not yet available")
            return None

    def vpa(self, pod_name, lower_bound, higher_bound, target):
        """scale that pod based-on the lower_bound, higher_bound and target
        it does the following:
        1. extracts the pod resource usage
        2. check loweer and higher bound against resource usage
        3. autoscales if necessary based-on the resource usage
             3.1. delete the previous pod
             3.2. starts the new pod
        """
        # TODO implement
        # self.monitor_pod()
        # do the checks
        # self.delete_pod()
        # self.create_pod()
        pass

    # --------------- properties ---------------

    @property
    def existing_pods(self):
        """returns available pods in the self.namespace"""
        try:
            existing_pods = list(
                map(
                    lambda pod: pod.metadata.name,
                    self._core_api.list_namespaced_pod(self.namespace).items,
                )
            )
        except ApiException as e:
            logger.error(e)
        return existing_pods

    @property
    def existing_deployments(self):
        """returns available deployments in the self.namespace"""
        try:
            existing_deployments = list(
                map(
                    lambda deployment: deployment.metadata.name,
                    self._apps_api.list_namespaced_deployment(
                        self.namespace).items,
                )
            )
        except ApiException as e:
            logger.error(e)
        return existing_deployments

    @property
    def existing_vpas(self):
        """returns vpas with their binded deployments
        Returns:
            {vpa_name: deployment_name}
        """
        response = self._objects_api.list_namespaced_custom_object(
            group="autoscaling.k8s.io",
            version="v1",
            namespace=self.namespace,
            plural="verticalpodautoscalers",
        )
        existing_vpas = dict(
            map(
                lambda a: (
                    a["spec"]["targetRef"]["name"], a["metadata"]["name"]),
                response["items"],
            )
        )
        return existing_vpas

    # --------------- utils ---------------

    def get_deployment_object(self, deployment_name):
        try:
            deployment_objs = self._apps_api.list_namespaced_deployment(
                self.namespace
            ).items
            deployment_obj = list(
                filter(
                    lambda a: a.metadata.name == deployment_name,
                    deployment_objs)
            )[0]
        except ApiException as e:
            logger.error(e)
        return deployment_obj

    def copy_file_inside_pod(
            self, pod_name: str, arcname: str, src_path: str, dest_path: str,
            namespace=None):
        """This function copies a file inside the pod
        :param pod_name: pod name
        :param arcname: actual name of file
        :param src_path: Source path of the file to be copied from
        :param dest_path: Destination path of the file to be copied into
        :param namespace: pod namespace
        """

        if namespace is None:
            # set default value for namespace
            namespace = self.namespace

        try:
            exec_command = ['tar', 'xvf', '-', '-C', dest_path]
            api_response = stream.stream(
                self._core_api.connect_get_namespaced_pod_exec, pod_name,
                namespace,
                command=exec_command,
                stderr=True,
                stdin=True,
                stdout=True,
                tty=False,
                _preload_content=False
            )

            logger.info('Uploading file "{}" to "{}" ...'.format(
                src_path, pod_name
            ))
            with TemporaryFile() as tar_buffer:
                with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                    tar.add(name=src_path, arcname=arcname)

                tar_buffer.seek(0)
                commands = [tar_buffer.read()]
                while api_response.is_open():
                    api_response.update(timeout=10)
                    if api_response.peek_stdout():
                        logger.info(
                            "uploading file %s successful." %
                            api_response.read_stdout())
                    if api_response.peek_stderr():
                        logger.error(
                            "uploading file %s failed." %
                            api_response.read_stderr())
                    if commands:
                        c = commands.pop(0)
                        api_response.write_stdin(c)
                    else:
                        break
                api_response.close()
        except ApiException as e:
            logger.error(
                "Exception when copying file to the pod: {}".format(e))
            self.clean(self.namespace)
            exit(1)
