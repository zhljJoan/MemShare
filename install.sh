#!/bin/bash

mkdir -p build && cd build
cmake .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
<<<<<<< HEAD
    -DCMAKE_PREFIX_PATH="/home/zlj/.miniconda3/envs/dgnn/lib/python3.10/site-packages" \
    -DPython3_ROOT_DIR="/home/zlj/.miniconda3/envs/dgnn" \
    -DCUDA_TOOLKIT_ROOT_DIR="/home/zlj/local/cuda-12.2" \
=======
    -DCMAKE_PREFIX_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    -DPython3_ROOT_DIR=$(python -c "import sys; print(sys.prefix)") \
    -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME:-"$(realpath $(dirname $(which nvcc))/../)"} \
>>>>>>> 98ad16d40e5e3e0a7dcdbc4f21dc9e164abc625f
&& make -j32 \
&& rm -rf ../starrygl/lib \
&& mkdir ../starrygl/lib \
&& cp lib*.so ../starrygl/lib/ \
&& patchelf --set-rpath '$ORIGIN:$ORIGIN/lib' --force-rpath ../starrygl/lib/*.so