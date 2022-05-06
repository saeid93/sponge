# Structure

This repository contains the code for our upcoming ML inference pipeline autoconfiguring project.

**Project Summary**:
Dataflow and computational pipelines have a longstanding history in the field of computing. ML inference is becoming one of the important factors in ML production systems life cycle. Many scenarios in production machine learning inference systems consist of more than one step for multiple steps of a machine learning inference scenario e.g. prepossessing, feature engineering, inference, and post-processing. ML pipelines may also represent a DAG like structure e.g several inference on multiple trained models could be aggregated into an ensemble stage or data sources of input data might be from multiple sources. A good ML inference pipelining system must be able to find the optimal resource and configuration assignment to the nodes of each node of the dataflow pipeline in a way that met both performance metrics like tail latency SLO and qualitative metrics like end to end accuracy of the inference pipeline subject to the users limited budget. Also incoming workloads from the users in production are variable and the initial configuration/resource assignment should be able to adapt to the fluctuating workloads in the cloud. In this work, we have proposed a solution for optimizing both qualitative and performance metrics jointly.


[Google Doc - Notes](https://docs.google.com/document/d/1VbMDl_09n77NCRk58C9vqzDLGkgfliPUYxS3NVX8fgw/edit?usp=sharing) \
[Paper Draft](https://www.overleaf.com/project/625456ee961f16abadd71f36)

# Physical Resource
* [Chameleon Cloud](https://chameleoncloud.org/)

# Technology Stack
## Setup

Do the steps in the following orders to setup the environment:
* **Python Environment**
  1. Download source code from GitHub
     ```
      git clone https://github.com/saeid93/infernece-pipeline-joint-optimization.git
     ```
  2. Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
  3. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
     ```
      conda create --name myenv python=3
     ```
  4. Activate conda environment
     ```
      conda activate myenv
     ```
  5. if you want to use GPUs make sure that you have the correct version of CUDA and cuDNN installed from [here](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
  6. Use [PyTorch](https://pytorch.org/) or [Tensorflow](https://www.tensorflow.org/install/pip#virtual-environment-install) isntallation manual to install one of them based-on your preference

  7. Install the followings
     ```
      sudo apt install cmake libz-dev
     ```
  8. Install requirements
     ```
      pip install -r requirements.txt
     ```
* **Infrastracture** [Kubernetes](https://kubernetes.io/)
   1. Install [Helm](https://helm.sh/docs/intro/install/)
   2. Lease a server from Chameleon cloud [chameleon-lease](docs/chameleon-lease.md)
   3. Setup a K8S cluster [k8s-setup](docs/setup-chameleon-k8s.md)
* **Network service mesh Tool** [Istio](https://istio.io/)
   1. Setup Istio on Chameleon [istio-setup](docs/setup-istio.md)
* **ML Inference DAG Technology** [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/)
   1. Setup the Seldon core operator on your cluster [seldon-core-installation](docs/setup-seldon-core-installation.md)
   2. See [Overview of Component](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/overview.html#metrics-with-prometheus) for an overview of the Seldon core framework
   3. Also see the link to the [shortlisted](docs/guide-seldon.md) parts of the documentation
* **Testing installation**
   1. Up to this point you should have a complete working installation
   2. To test the endpoints use [test-endpoints](seldon-core-examples/capabilities/test-endpoints/server_examples.ipynb)
   3. There are two options in Seldon for accessing a model endpint 1. [seldon-core-protocal](https://docs.seldon.io/projects/seldon-core/en/latest/reference/apis/index.html) and 2.https://kserve.github.io/website/modelserving/inference_api/ protocal. See them for the addresses of the endpoints exposed via the Istio gateway. The endpoints are also available through [Swagger API](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/serving.html#generated-documentation-swagger-ui) 
   4. To test use the following [istio-canary-example-notebook](seldon-core-examples/capabilities/istio/canary/istio_canary.ipynb)
   5. Make sure all the componentes of Seldon core are working. ❗ Make sure to port forward as instructed in the notebook.
* **Resources Observibility Tool** [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/)
   1. Setup the observibitliy tools for services resource usage monitoring [setup-observibility](docs/setup-prometeus-monitoring.md)
* **Network observibility Tool** [Istio](https://istio.io/)
   1. Setup Jeager on Chameleon [jeager-setup](docs/)
* **Load Generation Tool** [vegeta](https://github.com/tsenart/vegeta)
   1. [Guide to setup on Chameleon K8S cluster](ddd)
* **Docker and s2i**
   1. For some of the pipeline you'll need [Dcoker](https://www.docker.com/) and [s2i](https://github.com/openshift/source-to-image)
   2. Install them using the offical documentation for [docker-doc](https://docs.docker.com/engine/install/ubuntu/) and [s2i-doc](https://github.com/openshift/source-to-image#installation)
* **Minio and nfs**
   1. [Minio](https://min.io/) and [nfs](https://en.wikipedia.org/wiki/Network_File_System) are needed for the storage
   2. Setup them using [guide-storage](docs/guide-storage.md)
* **OpenStack**
   1. TODO
* **Pipelines**
   1. [Medium Article Example](https://becominghuman.ai/seldon-inference-graph-pipelined-model-serving-211c6b095f62), in [1-example-pipeline](pipelines/1-example-pipeline): Good example of TODO complete
* **Guide to Deploy a model and pipeline**
   1. [Guide-model-deployment](docs/guide-model-deployment.md)
* 🔴 Add other as you go

* **Common Problems**
   1. List of the [common problems](docs/common-problems.md) and bugs


TODO 🔴 automate all the above steps


# Pipelines


## Guide
* **Istio**
TODO
* **Prometheus**
TODO


