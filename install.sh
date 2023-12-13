#!/bin/bash

mkdir -p build && cd build
cmake .. \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_PREFIX_PATH="/home/hwj/.miniconda3/envs/sgl/lib/python3.10/site-packages" \
    -DPython3_ROOT_DIR="/home/hwj/.miniconda3/envs/sgl" \
    -DCUDA_TOOLKIT_ROOT_DIR="/home/hwj/.local/cuda-11.7" \
&& make -j32 \
&& cp libstarrygl_ops.so ../starrygl/lib/