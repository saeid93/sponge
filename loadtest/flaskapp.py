from flask import Flask
from flask import request

app = Flask(__name__)


@app.route('/')
def hello():
    return "hello"


def process_data(data):
    x = data["value"]
    return x*2


@app.route('/prediction', methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        data = request.json
        return_value = process_data(data)
        return {"data" : return_value}
    return "post data pleas" \
           "e"

