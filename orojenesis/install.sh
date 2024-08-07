#!/bin/bash
sudo apt install -y scons libconfig++-dev libboost-dev libboost-iostreams-dev libboost-serialization-dev libyaml-cpp-dev libncurses-dev libtinfo-dev libgpm-dev git build-essential python3-pip jupyter-core

pushd ../
pushd ./src && ln -s ../pat-public/src/pat . && popd
scons --static -j4
export TIMELOOP_BASE_PATH=$(pwd)

popd && pip install -r requirements.txt
