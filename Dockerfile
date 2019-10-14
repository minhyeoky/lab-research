FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get install -y libsndfile1 vim

COPY ./requirements.txt /
RUN pip install -r /requirements.txt
