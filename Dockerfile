FROM tensorflow/tensorflow:latest-gpu-py3

COPY ./requirements.txt /

RUN apt-get install -y libsndfile1
RUN pip install -r /requirements.txt
RUN apt-get install -y vim