#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/pfb64/CUDA/out/ && \
rm -rf -- ~/build/mnp/pfb64/cuda && \
mkdir -p ~/build/mnp/pfb64/cuda && \

# run cmake
cd ~/build/mnp/pfb64/cuda && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=g++ ${source_folder} && \

make pfb64_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
