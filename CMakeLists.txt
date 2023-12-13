cmake_minimum_required(VERSION 3.15)

project(starrygl_ops VERSION 0.1)

option(WITH_PYTHON "Link to Python when building" ON)
option(WITH_CUDA "Link to CUDA when building" ON)
option(WITH_METIS "Link to METIS when building" ON)
option(WITH_MTMETIS "Link to multi-threaded METIS when building" ON)

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
    add_definitions(-DWITH_METIS)
    set(GKLIB_DIR "${CMAKE_SOURCE_DIR}/third_party/GKlib")
    set(METIS_DIR "${CMAKE_SOURCE_DIR}/third_party/METIS")

    set(GKLIB_INCLUDE_DIRS "${GKLIB_DIR}/include")
    set(GKLIB_LIBRARIES "${GKLIB_DIR}/lib")

    set(METIS_INCLUDE_DIRS "${METIS_DIR}/include")
    set(METIS_LIBRARIES "${METIS_DIR}/lib")

    include_directories(${METIS_INCLUDE_DIRS})
    link_libraries(${METIS_LIBRARIES})
endif()

if(WITH_MTMETIS)
    add_definitions(-DWITH_MTMETIS)
    set(MTMETIS_DIR "${CMAKE_SOURCE_DIR}/third_party/mt-metis")

    set(MTMETIS_INCLUDE_DIRS "${MTMETIS_DIR}/include")
    set(MTMETIS_LIBRARIES "${MTMETIS_DIR}/lib")

    include_directories(${MTMETIS_INCLUDE_DIRS})
    link_libraries(${MTMETIS_LIBRARIES})
endif()

find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
add_compile_options(${TORCH_CXX_FLAGS})

include_directories("csrc/include")

file(GLOB_RECURSE UVM_SRCS "csrc/uvm/*.cpp")
file(GLOB_RECURSE METIS_SRCS "csrc/uvm/*.cpp")

add_library(${PROJECT_NAME} SHARED csrc/export.cpp ${UVM_SRCS} ${METIS_SRCS})

if(WITH_PYTHON)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE -DTORCH_EXTENSION_NAME=lib${PROJECT_NAME})

# set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "_C")
# install(TARGETS ${PROJECT_NAME} DESTINATION  "${CMAKE_SOURCE_DIR}/starrygl/lib")
