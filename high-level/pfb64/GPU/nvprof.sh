#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/pfb64/GPU/out/ && \
rm -rf -- ~/build/mnp/pfb64/gpu && \
mkdir -p ~/build/mnp/pfb64/gpu && \

# run cmake
cd ~/build/mnp/pfb64/gpu && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Nvprof -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make pfb64_0 && \
cd ${source_folder} && \

sbatch nvprof-job.sh
