# **Text Analysis**

---

#### Build

```
$ s2i build . seldonio/seldon-core-s2i-python3:1.13.0-dev urthilak/text-tagging:0.1.5
$ docker push urthilak/text-tagging:0.1.5
```

#### Test

```
$ docker run --name "text-tagging" --rm urthilak/text-tagging:0.1.5
$ docker exec -it text-tagging python TextTagging_Test.py
```

#### Usage

```
$ docker run --name "text-tagging" --rm -p 5001:5000 urthilak/text-tagging:0.1.5
$ curl -g http://localhost:5001/predict --data-urlencode 'json={"data": {"names": ["message"], "ndarray": ["The pigeonhole principle is a simple, yet beautiful and useful idea. Given a set A of pigeons and a set B of pigeonholes, if all the pigeons fly into a pigeonhole and there are more pigeons than holes, then one of the pigeonholes has to contain more than one pigeon."]}, "meta": {"tags":{"sentiment_analysis_passed":true}}}'
```

#### Deploy to test (Optional)
```angular2html
kubectl apply -f - << END
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: tagging17
  namespace: seldon
spec:
  name: tagging17
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: urthilak/text-tagging:0.1.8
          name: classifier
          env:
          - name: FLASK_SINGLE_THREADED
            value: '1'
        terminationGracePeriodSeconds: 1
    graph:
      children: []
      endpoint:
        type: REST
      name: classifier
      type: MODEL
    labels:
      version: v1
    name: default
    replicas: 1
END
```