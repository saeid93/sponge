FROM python:latest
RUN pip install requests
RUN pip install kopf
RUN pip install kubernetes
COPY main.py /main.py
CMD kopf run ./main.py --verbose

