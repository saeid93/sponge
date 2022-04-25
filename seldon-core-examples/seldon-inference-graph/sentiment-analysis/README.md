# **Sentiment Analysis**

---

#### Build

```
$ s2i build . seldonio/seldon-core-s2i-python3 urthilak/sentiment-analysis:0.1.0
$ docker push urthilak/sentiment-analysis:0.1.0
```

#### Test

```
$ docker run --name "sentiment-analysis" --rm urthilak/sentiment-analysis:0.1.0
$ docker exec -it sentiment-analysis python SentimentAnalysis_Test.py
```

#### Usage

```
$ docker run --name "sentiment-analysis" --rm -p 5001:5000 urthilak/sentiment-analysis:0.1.0
$ curl -g http://localhost:5001/predict --data-urlencode 'json={"data": {"names": ["message"], "ndarray": ["All too much of the man-made is ugly, inefficient, depressing chaos."]}}'
```
#### Deploy to test (Optional)

```
kubectl apply -f - << END
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: sentiment-analysis11
  namespace: seldon
spec:
  name: sentiment-analysis11
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: urthilak/sentiment-analysis:0.1.11
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

