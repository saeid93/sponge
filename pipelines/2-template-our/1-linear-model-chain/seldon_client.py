# example of connecting to Seldon with Python
# instead of curl
import numpy as np
from seldon_core.seldon_client import SeldonClient

deployment_name = "openvino-model"
namespace = "default"
X = np.array([1,3,2,5])

sc = SeldonClient(deployment_name=deployment_name, namespace=namespace)

response = sc.predict(
    gateway_endpoint="localhost:32000",gateway="istio", transport="grpc", data=X, client_return_type="proto"
)

result = response.response.data.tensor.values
