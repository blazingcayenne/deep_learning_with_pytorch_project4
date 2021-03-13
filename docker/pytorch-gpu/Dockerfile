FROM mhs-base-pytorch:latest
LABEL maintainer="kevin@minnhealth.com"

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y unzip
RUN apt-get install -y wget

#RUN /root/miniconda/bin/pip install -U catalyst
RUN /root/miniconda/bin/pip install -U albumentations
RUN /root/miniconda/bin/pip install git+https://github.com/qubvel/segmentation_models.pytorch

#RUN /root/miniconda/bin/pip install -U segmentation-models-pytorch
#RUN /root/miniconda/bin/pip install detectron2 -f \
#  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

COPY adds/entrypoint.py /entrypoint.py
COPY adds/matplotlibrc /matplotlibrc

ENV PATH=/root/miniconda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV MATPLOTLIBRC=/

ENTRYPOINT ["/entrypoint.py"]
