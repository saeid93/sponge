# **Summarize Text**

---

#### Build

```
$ s2i build . seldonio/seldon-core-s2i-python:3.0.7 sdghafouri/nlp:node-2-summarize-text
$ docker push sdghafouri/nlp:node-2-summarize-text
```

#### Test

```
$ docker run --name "summarize-text" --rm sdghafouri/nlp:node-2-summarize-text
$ docker exec -it summarize-text python SummarizeText_Test.py
```

#### Usage

```
$ docker run --name "summarize-text" --rm -p 5001:5000 sdghafouri/nlp:node-2-summarize-text
$ curl -g http://localhost:5001/predict --data-urlencode 'json={"data": {"names": ["message"], "ndarray": ["In this paper we present Katecheo, a portable and modular system for reading comprehension based question answering that attempts to ease this development burden. The system provides a quickly deployable and easily extendable way for developers to integrate question answering functionality into their applications. Katecheo includes four configurable modules that collectively enable identification of questions, classification of those questions into topics, a search of knowledge base articles, and reading comprehension. The modules are tied together in a single inference graph that can be invoked via a REST API call"]}, "meta": {"tags":{"text_tagging_passed":true}}}'
```
