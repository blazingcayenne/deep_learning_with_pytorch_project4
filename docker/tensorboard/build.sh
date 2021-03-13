#!/bin/bash

if [ $# -lt 1 ]; then
    IMAGE_NAME=mhs-base-tensorboard
else
    IMAGE_NAME=$1
fi

sudo docker build --rm -t ${IMAGE_NAME} .
