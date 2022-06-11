# Structure

This repository contains the code for our upcoming ML inference pipeline autoconfiguring projects.

**Project 1 Summary** [What you see should not be your Backend](https://www.overleaf.com/read/pvmxxqcgcfnn):
TODO pending the main document

**Project 2 Summary** [Joint Optimization of Qualitative and Performance Metrics on ML inference DAGs](https://www.overleaf.com/read/pfnwptxyktff):

Dataflow and computational pipelines have a longstanding history in the field of computing. ML inference is becoming one of the important factors in ML production systems life cycle. Many scenarios in production machine learning inference systems consist of more than one step for multiple steps of a machine learning inference scenario e.g. prepossessing, feature engineering, inference, and post-processing. ML pipelines may also represent a DAG like structure e.g several inference on multiple trained models could be aggregated into an ensemble stage or data sources of input data might be from multiple sources. A good ML inference pipelining system must be able to find the optimal resource and configuration assignment to the nodes of each node of the dataflow pipeline in a way that met both performance metrics like tail latency SLO and qualitative metrics like end to end accuracy of the inference pipeline subject to the users limited budget. Also incoming workloads from the users in production are variable and the initial configuration/resource assignment should be able to adapt to the fluctuating workloads in the cloud. In this work, we have proposed a solution for optimizing both qualitative and performance metrics jointly.

**Project 3 Summary** TODO Mehran's project

[Google Doc - Notes](https://docs.google.com/document/d/1VbMDl_09n77NCRk58C9vqzDLGkgfliPUYxS3NVX8fgw/edit?usp=sharing)

1. Download subproject source code from GitHub
   ```
   git clone https://github.com/saeid93/infernece-pipeline-joint-optimization.git
   ```
   > [Optional] To get contents of the master_project, run:
   > ```shell
   > git submodule init
   > git submodule update
   > ```

For setting infrastructure refer to accompanying repo [infrastructure](https://github.com/reconfigurable-ml-pipeline/infrastructure)