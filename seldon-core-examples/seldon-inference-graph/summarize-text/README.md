# **Summarize Text**

---

#### Build

```
$ s2i build . seldonio/seldon-core-s2i-python3:1.13.0-dev urthilak/summarize-text:0.1.8
$ docker push urthilak/summarize-text:0.1.8
```

#### Test

```
$ docker run --name "summarize-text" --rm urthilak/summarize-text:0.1.8
$ docker exec -it summarize-text python SummarizeText_Test.py
```

#### Usage

```
$ docker run --name "summarize-text" --rm -p 5001:5000 urthilak/summarize-text:0.1.8
$ curl -g http://localhost:5001/predict --data-urlencode 'json={"data": {"names": ["message"], "ndarray": ["In this paper we present Katecheo, a portable and modular system for reading comprehension based question answering that attempts to ease this development burden. The system provides a quickly deployable and easily extendable way for developers to integrate question answering functionality into their applications. Katecheo includes four configurable modules that collectively enable identification of questions, classification of those questions into topics, a search of knowledge base articles, and reading comprehension. The modules are tied together in a single inference graph that can be invoked via a REST API call"]}, "meta": {"tags":{"text_tagging_passed":true}}}'

```


#### Deploy to test (Optional)
```angular2html
kubectl apply -f - << END
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: summarize15
  namespace: seldon
spec:
  name: summarize15
  predictors:
  - componentSpecs:
    - spec:
        containers:
        - image: urthilak/summarize-text:0.1.5
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