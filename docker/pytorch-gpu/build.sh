#!/bin/bash

if [ $# -lt 1 ]; then
    IMAGE_NAME=mhs-pytorch-gpu-dlwp
else
    IMAGE_NAME=$1
fi

docker build --rm -t ${IMAGE_NAME} .
