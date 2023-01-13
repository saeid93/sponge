FROM python:3.6

RUN mkdir -p /GRPC/code
RUN mkdir -p /data/persons
RUN pip3 install --user --upgrade h5py==2.10.0 tensorflow==2.3.0
RUN pip3 install --user --upgrade numpy grpcio keras Pillow matplotlib keras_vggface keras_applications keras_preprocessing requests psutil
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6'  -y

COPY main.py /app/main.py
COPY input-sample.JPEG /app/input-sample.JPEG

WORKDIR /app

ENV PYTHONUNBUFFERED=0

ENTRYPOINT ["python", "-u", "main.py"]
