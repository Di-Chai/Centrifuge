FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive

# Temporary Installation Directory
ENV STAGE_DIR=/tmp
RUN mkdir -p ${STAGE_DIR}

# Installation/Basic Utilities
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        cmake g++ gcc \
        curl wget vim tmux emacs less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools \
        llvm-dev debhelper dh-virtualenv

ENV HF_HOME=/HuggingFace
ENV PATH=$PATH:~/.local/bin

# For fixing the dependency conflict between dill and datasets
RUN sed -i '/^dill==/d' /etc/pip/constraint.txt

RUN pip install huggingface_hub transformers==4.45.2 datasets==4.0.0 requests>=2.32.2 accelerate==1.0.0 peft==0.15.2 tiktoken nltk torchao==0.12.0 torchtune

# Vis
RUN pip install torchviz matplotlib

# DeepSpeed
RUN pip install deepspeed==0.17.1 && ds_report

# Megatron-LM
RUN pip install megatron-core

RUN apt install -y graphviz