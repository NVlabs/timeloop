FROM ubuntu:20.04 AS builder

ENV BUILD_DIR=/usr/local/src

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
    && apt-get install -y --no-install-recommends locales \
    && apt-get install -y --no-install-recommends git \
    && apt-get install -y --no-install-recommends scons \
    && apt-get install -y --no-install-recommends make \
    && apt-get install -y --no-install-recommends python3.10 \
    && apt-get install -y --no-install-recommends python3-pip \
    && apt-get install -y --no-install-recommends doxygen \
    && apt-get install -y --no-install-recommends unzip \
    && apt-get install -y --no-install-recommends zip \
    && rm -rf /var/lib/apt/lists/* \
    && if [ ! -d $BUILD_DIR ]; then mkdir $BUILD_DIR; fi


# Build and install timeloop dependencies
WORKDIR $BUILD_DIR

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
    && apt-get install -y --no-install-recommends \
                       locales \
                       curl \
                       git \
                       wget \
                       python3-dev \
                       python3-pip \
                       scons \
                       make \
                       autotools-dev \
                       autoconf \
                       automake \
                       libtool \
                       vim \
    && apt-get install -y --no-install-recommends \
                       g++ \
                       cmake

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
                       g++ \
                       libconfig++-dev \
                       libboost-dev \
                       libboost-iostreams-dev \
                       libboost-serialization-dev \
                       libyaml-cpp-dev \
                       libncurses5-dev \
                       libtinfo-dev \
                       libgpm-dev \
                       libgmp-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir notebook

# export port 8888
EXPOSE 8888


WORKDIR /home/workspace
CMD ["bash"]

RUN git clone --recurse-submodules  https://github.com/NVlabs/timeloop.git
RUN cd timeloop && git checkout oaves_keep_max &&   \
    cd ./src && ln -s ../pat-public/src/pat . &&  cd .. && \
    scons --static -j4 && \
    export TIMELOOP_BASE_PATH=$(pwd)

RUN pip install --no-cache-dir \
numpy==1.24.3 \
pandas==2.0.1 \
typing_extensions==4.5.0 \
PyYAML \
networkx==3.1 \
notebook==6.5.4 \
more-itertools==8.10.0 \
seaborn==0.12.2 \
fire==0.5.0
