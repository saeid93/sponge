# **Sentiment Analysis**

---

#### Build

```
$ s2i build . seldonio/seldon-core-s2i-python3:0.7 sdghafouri/nlp:node-1-sentiment-analysis
$ docker push sdghafouri/nlp:node-1-sentiment-analysis
```

#### Test

```
$ docker run --name "sentiment-analysis" --rm sdghafouri/nlp:node-1-sentiment-analysis
$ docker exec -it sentiment-analysis python SentimentAnalysis_Test.py
```

#### Usage

```
$ docker run --name "sentiment-analysis" --rm -p 5001:5000 sdghafouri/nlp:node-1-sentiment-analysis
$ curl -g http://localhost:5001/predict --data-urlencode 'json={"data": {"names": ["message"], "ndarray": ["All too much of the man-made is ugly, inefficient, depressing chaos."]}}'
```