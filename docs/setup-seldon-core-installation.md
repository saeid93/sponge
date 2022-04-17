# Seldon core isntallation

1. Seldon is a framework for making complex grpc and rest apis for the trained ML models
2. Install the istio verstion from [istio-documentation](https://istio.io/latest/docs/setup/getting-started/) and make sure it is running on the server and make sure that it is running the sample application
3. Install it easily using helm like the documentation from this notebook [seldon_core_setup.ipynb](seldon-core/notebooks/seldon_core_setup.ipynb)
4. Install the ingress for istio (not the ssl one)
5. Related notebooks
    1. [server_examples.ipynb](seldon-core/notebooks/server_examples.ipynb): build up a simple sklearn model from a pretrained model