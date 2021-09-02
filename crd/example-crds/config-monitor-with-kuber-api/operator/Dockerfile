FROM python:latest
RUN pip install requests
RUN pip install kubernetes
COPY main.py /main.py
ENTRYPOINT [ "python", "/main.py" ]
