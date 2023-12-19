#!/bin/sh
#conda activate gnn
cd ./starrygl/sample/sample_core
if [ -f "setup.py" ]; then
    rm -r build
    rm sample_cores.cpython-*.so
    python setup.py build_ext --inplace
fi
cd ../part_utils
if [  -f "setup.py" ]; then
    rm -r build
    rm torch_utils.cpython-*.so
    python setup.py build_ext --inplace
fi
cd ../../
