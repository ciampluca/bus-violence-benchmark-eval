#!/bin/bash

BUS_VIOLENCE_DIR="bus-violence"
RLV_DIR="real-life-violence"
RWF_DIR="RWF-2000"
SCF_DIR="surveillance-camera-fight"

RLV_URL="https://drive.google.com/uc?id=1zw-elBkWVaGN1qKxXeKZsau_8iep9wl1"
RWF_URL="https://drive.google.com/uc?id=1xNXdpqJ7-Jd2aWI1InHh2-Zfob8hYUuk"
SCF_URL="https://drive.google.com/uc?id=1aYAP6iL80j1_d0L6zAdUOoVx563cgtwU"


# BUS-VIOLENCE DATASET
if [[ ! -d "${BUS_VIOLENCE_DIR}" ]]; then
    echo "Downloading and extracting: bus-violence"
    zenodo_get 10.5281/zenodo.7044203       # zenodo DOI
    unzip bus-violence.zip -d "${BUS_VIOLENCE_DIR}"  && rm bus-violence.zip && rm md5sums.txt
fi

# REAL-LIFE VIOLENCE SITUATIONS DATASET
if [[ ! -d "${RLV_DIR}" ]]; then
    echo "Downloading and extracting: real-life violence situations"
    gdown "${RLV_URL}"
    unzip real-life-violence.zip && rm real-life-violence.zip
fi

# RWF-2000 DATASET
if [[ ! -d "${RWF_DIR}" ]]; then
    echo "Downloading and extracting: RWF-2000"
    gdown "${RWF_URL}"
    unzip RWF-2000.zip && rm RWF-2000.zip
fi

# SURVEILLANCE CAMERA FIGHT DATASET
if [[ ! -d "${SCF_DIR}" ]]; then
    echo "Downloading and extracting: surveillance camera fight"
    gdown "${SCF_URL}"
    unzip surveillance-camera-fight.zip && rm surveillance-camera-fight.zip
fi