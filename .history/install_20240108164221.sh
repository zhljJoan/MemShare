#!/bin/bash

mkdir -p build && cd build
cmake .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_PREFIX_PATH="/home/zlj/.miniconda3/envs/dgnn/lib/python3.8/site-packages" \
    -DPython3_ROOT_DIR="/home/zlj/.miniconda3/envs/dgnn" \
    -DCUDA_TOOLKIT_ROOT_DIR="/home/zlj/local/cuda-12.2" \
&& make -j32 \
&& rm -rf ../starrygl/lib \
&& mkdir ../starrygl/lib \
&& cp lib*.so ../starrygl/lib/ \
&& patchelf --set-rpath '$ORIGIN:$ORIGIN/lib' --force-rpath ../starrygl/lib/*.so
