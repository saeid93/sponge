FROM pytorch/pytorch:latest

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY main.py /app/main.py
COPY input-sample.JPEG /app/input-sample.JPEG

WORKDIR /app

ENV PYTHONUNBUFFERED=0

ENTRYPOINT ["python", "-u", "main.py"]
