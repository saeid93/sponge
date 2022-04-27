FROM python:3.7-slim
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
EXPOSE 9000

ENV MODEL_NAME node.NodeThree
ENV SERVICE_TYPE MODEL
ENV FLASK_DEBUG True

RUN chown -R 8888 /app

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE