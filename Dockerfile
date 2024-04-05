# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved. 

# To build the docker container, run: $  docker build -f Dockerfile -t zhangkin/nways .
# To run: $ docker run --gpus all --rm -it -p 8888:8888 zhangkin/nways
# Finally, open http://localhost:8888/

FROM nvcr.io/nvidia/nvhpc:23.5-devel-cuda_multi-ubuntu20.04

RUN apt-get -y update && \
        DEBIAN_FRONTEND=noninteractive apt-get -yq install --no-install-recommends python3-pip python3-setuptools nginx zip make build-essential libtbb-dev && \
        rm -rf /var/lib/apt/lists/* && \
        pip3 install --upgrade pip &&\
        pip3 install gdown
        
RUN apt-get update -y        
RUN apt-get install -y git nvidia-modprobe kmod
RUN pip3 install jupyterlab
RUN pip3 install ipywidgets

############################################
RUN apt-get update -y

# TO COPY the data
COPY _basic/openacc/ /root/labs/openacc
COPY _basic/openmp/ /root/labs/openmp
COPY _basic/_common/ /root/labs/_common
COPY _basic/iso/ /root/labs/iso
COPY _basic/cuda/ /root/labs/cuda
COPY _basic/_start_nways.ipynb /root/labs

COPY _challenge/C/jupyter_notebook/ /root/challenge/C/jupyter_notebook
COPY _challenge/C/source_code/ /root/challenge/C/source_code
COPY _challenge/minicfd.ipynb /root/challenge

RUN python3 /root/labs/_common/dataset.py
#################################################
ENV PATH="/usr/local/bin:/opt/anaconda3/bin:/usr/bin:$PATH"
#################################################

#ADD nways_labs/ /root
WORKDIR /root
CMD jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/root
