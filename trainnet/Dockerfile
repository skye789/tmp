FROM nvcr.io/nvidia/pytorch:21.08-py3


RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common


# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing;Ampere"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN apt-get clean && apt-get update && apt-get install -y \
    locales \
    language-pack-fi  \
    language-pack-en && \
    export LANGUAGE=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    export LC_ALL=en_US.UTF-8 && \
    locale-gen en_US.UTF-8 && \
    dpkg-reconfigure locales

RUN add-apt-repository ppa:savoury1/ffmpeg4
RUN apt-get install -y --no-install-recommends pkg-config
RUN apt install -y --no-install-recommends ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev
    
RUN apt-get update && apt-get install -y \
	cmake ninja-build protobuf-compiler libprotobuf-dev libmagickwand-dev && \
  rm -rf /var/lib/apt/lists/*

RUN ln -sv /usr/bin/python3 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	 python3 get-pip.py && \
	 rm get-pip.py

COPY data /workspace/Trainnet/data
COPY pcdet /workspace/Trainnet/pcdet
COPY tools /workspace/Trainnet/tools
COPY setup.py /workspace/Trainnet/setup.py

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install

RUN pip install -r requirements.txt

RUN cd Trainnet && python setup.py develop


