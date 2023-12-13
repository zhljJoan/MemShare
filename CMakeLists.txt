cmake_minimum_required(VERSION 3.15)

project(starrygl_ops VERSION 0.1)

option(WITH_PYTHON "Link to Python when building" ON)
option(WITH_CUDA "Link to CUDA when building" ON)
option(WITH_METIS "Link to METIS when building" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(WITH_PYTHON)
    add_definitions(-DWITH_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    include_directories(${Python3_INCLUDE_DIRS})
endif()

if(WITH_CUDA)
    add_definitions(-DWITH_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")
endif()

if(WITH_METIS)
    set(GKLIB_DIR "third_party/GKlib")
    set(METIS_DIR "third_party/METIS")

    add_subdirectory(${GKLIB_DIR})
endif()

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
add_compile_options(${TORCH_CXX_FLAGS})

# find_package(OpenMP REQUIRED)

include_directories("csrc/include")

file(GLOB_RECURSE UVM_SRCS "csrc/uvm/*.cpp")
file(GLOB_RECURSE METIS_SRCS "csrc/uvm/*.cpp")

add_library(${PROJECT_NAME} SHARED csrc/export.cpp ${UVM_SRCS} ${METIS_SRCS})

if(WITH_PYTHON)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
# target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

target_compile_definitions(${PROJECT_NAME} PRIVATE -DTORCH_EXTENSION_NAME=lib${PROJECT_NAME})

# set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "_C")
install(TARGETS ${PROJECT_NAME} DESTINATION  "${CMAKE_SOURCE_DIR}/starrygl/lib")
