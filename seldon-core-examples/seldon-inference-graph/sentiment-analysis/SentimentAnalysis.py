from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalysis(object):
    result = {}
    analyser = None

    def __init__(self):
        self.analyser = SentimentIntensityAnalyzer()

    def predict(self, X, features_names, meta):
        if len(X) > 0:
            sentence = X[0]
            score = self.analyser.polarity_scores(sentence)

            self.result["sentiment_analysis_passed"] = True
            self.result["input_text"] = sentence
            self.result["sentiment_analysis_result"] = score
        else:
            self.result["sentiment_analysis_passed"] = False
            self.result["failure_reason"] = "Sentiment Analysis does not have a valid input array"
        return X

    def tags(self):
        return self.result

    def metrics(self):
        return [
            {"type": "COUNTER", "key": "mycounter", "value": 1}, # a counter which will increase by the given value
            {"type": "GAUGE", "key": "mygauge", "value": 100},   # a gauge which will be set to given value
            {"type": "TIMER", "key": "mytimer", "value": 20.2},  # a timer which will add sum and count metrics - assumed millisecs
        ]
