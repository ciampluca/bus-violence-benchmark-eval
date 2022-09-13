FROM docker.io/pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

RUN apt update && apt install -y build-essential wget unzip nano
RUN apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt .
RUN pip install -r requirements.txt