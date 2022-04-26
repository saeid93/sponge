FROM python:3.8-slim
WORKDIR /app

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy source code
COPY ClassificationModel.py .

# Port for REST
EXPOSE 9000

# Define environment variables
ENV MODEL_NAME ClassificationModel
ENV SERVICE_TYPE MODEL


CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE