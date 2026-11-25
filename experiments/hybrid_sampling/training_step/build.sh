#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
root=$(cd ../../.. && pwd)
include=$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')
torch_include=$(python -c 'from torch.utils.cpp_extension import include_paths; print(include_paths()[0])')
suffix=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
"${CXX:-c++}" -std=c++17 -O3 -shared -fPIC -I"$include" -I"$torch_include" \
    -I"$root/csrc/sampler/include" compact.cpp -o "memshare_packed_block$suffix"
