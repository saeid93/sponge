from barazmoon import MLServerBarAzmoon

endpoint: str = "http://localhost:32000/seldon/default/custom-mlserver/v2/models/infer"
http_method = 'post'
workload = [10, 7, 4, 12, 15]

load_Gen = MLServerBarAzmoon(endpoint, http_method, workload)
load_Gen.start()