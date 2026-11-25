#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
include=$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')
torch_include=$(python -c 'from torch.utils.cpp_extension import include_paths; print(include_paths()[0])')
suffix=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
"${CXX:-c++}" -O3 -std=c++17 -shared -fPIC -fopenmp -I"$include" -I"$torch_include" \
    -DTORCH_EXTENSION_NAME=memshare_hybrid_cpu cpu.cpp -o "memshare_hybrid_cpu$suffix"
