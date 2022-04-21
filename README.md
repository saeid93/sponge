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

* **Infrastracture** [Kubernetes](https://kubernetes.io/)
   *  Install [Helm](https://helm.sh/docs/intro/install/)
   *  Lease a server from Chameleon cloud [chameleon-lease](docs/chameleon-lease.md)
   *  Setup a K8S cluster [k8s-setup](docs/chameleon-k8s.md)
* **Network service mesh Tool** [Istio](https://istio.io/)
   * Setup Istio on Chameleon [istio-setup](docs/setup-istio.md)
* **ML Inference DAG Technology** [Seldon Core](https://docs.seldon.io/projects/seldon-core/en/latest/)
   * Setup the Seldon core operator on your cluster [seldon-core-installation](docs/setup-seldon-core-installation.md)
   * See [Overview of Component](https://docs.seldon.io/projects/seldon-core/en/latest/workflow/overview.html#metrics-with-prometheus) for an overview of the Seldon core framework
   * Also see the link to the [shortlisted](docs/seldon.md) parts of the documentation
* **Resources Observibility Tool** [Prometheus](https://prometheus.io/) and [Grafana](https://grafana.com/)
   * Setup the observibitliy tools for services resource usage monitoring [setup-observibility](docs/prometeus-monitoring.md)
* **Network observibility Tool** [Istio](https://istio.io/)
   * Setup Jeager on Chameleon [jeager-setup](docs/)
* **Load Generation Tool** [vegeta](https://github.com/tsenart/vegeta)
   * [Guide to setup on Chameleon K8S cluster](ddd)
* **Enable PVC on K8S for Model Storage**
   * [Enabling dashboards-TODO](ddd)
* **Docker and s2i**
   * For some of the pipeline you'll need [Dcoker](https://www.docker.com/) and [s2i](https://github.com/openshift/source-to-image)
   * Install them using the offical documentation for [docker-doc](https://docs.docker.com/engine/install/ubuntu/) and [s2i-doc](https://github.com/openshift/source-to-image#installation)
* 🔴 Add other as you go

TODO 🔴 automate all the above steps


# Pipelines


## Guide
* **Istio**
TODO
* **Prometheus**
TODO


