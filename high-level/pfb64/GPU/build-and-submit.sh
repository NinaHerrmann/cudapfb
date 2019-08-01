#!/bin/bash

source_folder=${PWD} && \

# remove files and create folder
mkdir -p /home/fwrede/musket-build/pfb64/GPU/out/ && \
rm -rf -- /home/fwrede/musket-build/pfb64/GPU/build/benchmark && \
mkdir -p /home/fwrede/musket-build/pfb64/GPU/build/benchmark && \

# run cmake
cd /home/fwrede/musket-build/pfb64/GPU/build/benchmark && \
cmake -G "Unix Makefiles" -D CMAKE_BUILD_TYPE=Benchmarktaurus -D CMAKE_CXX_COMPILER=pgc++ ${source_folder} && \

make pfb64_0 && \
cd ${source_folder} && \

sbatch job.sh
