FROM python:3.8.0

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY main.py /app/main.py
COPY input-sample.JPEG /app/input-sample.JPEG
COPY face-resnet50.h5 /app/face-resnet50.h5

WORKDIR /app

ENV PYTHONUNBUFFERED=0

ENTRYPOINT ["python", "-u", "main.py"]
