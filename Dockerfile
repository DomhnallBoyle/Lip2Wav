# FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    ffmpeg \
    yasm

WORKDIR /opt/lip2wav
COPY requirements.txt /opt/lip2wav

RUN python3 -m pip install --upgrade pip
RUN pip3 install Cython numpy
RUN pip3 install -r requirements.txt
