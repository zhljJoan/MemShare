#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-coordinated-sampler}"
CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_JOBS="${BUILD_JOBS:-4}"

TORCH_CMAKE_PREFIX="$(${PYTHON_BIN} -c \
    "import torch; print(torch.utils.cmake_prefix_path)")"
PYTHON_ROOT="$(${PYTHON_BIN} -c "import sys; print(sys.prefix)")"

"${CMAKE_BIN}" \
    -S "${REPO_ROOT}" \
    -B "${BUILD_DIR}" \
    -DWITH_CUDA=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}" \
    -DPython3_ROOT_DIR="${PYTHON_ROOT}"
"${CMAKE_BIN}" \
    --build "${BUILD_DIR}" \
    --target starrygl_sampler \
    -j"${BUILD_JOBS}"

SAMPLER_LIBRARY="${BUILD_DIR}/libstarrygl_sampler.so"
if [[ ! -f "${SAMPLER_LIBRARY}" ]]; then
    echo "sampler build completed but ${SAMPLER_LIBRARY} was not found" >&2
    exit 1
fi

echo "Built ${SAMPLER_LIBRARY}"
echo "Test with:"
echo "COORDINATED_SAMPLER_NATIVE_LIB=${SAMPLER_LIBRARY} python -m pytest -q tests/test_coordinated_sampler.py"
