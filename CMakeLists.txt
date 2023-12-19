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

find_package(OpenMP REQUIRED)
link_libraries(OpenMP::OpenMP_CXX)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
add_compile_options(${TORCH_CXX_FLAGS})

if(WITH_PYTHON)
    add_definitions(-DWITH_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    include_directories(${Python3_INCLUDE_DIRS})
endif()

if(WITH_CUDA)
    add_definitions(-DWITH_CUDA)
    add_definitions(-DWITH_UVM)

    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/lib64/libcudart.so")

    file(GLOB_RECURSE UVM_SRCS "csrc/uvm/*.cpp")

    add_library(uvm SHARED ${UVM_SRCS})
    target_link_libraries(uvm PRIVATE ${TORCH_LIBRARIES})
endif()

if(WITH_METIS)
    add_definitions(-DWITH_METIS)
    set(GKLIB_DIR "${CMAKE_SOURCE_DIR}/third_party/GKlib")
    set(METIS_DIR "${CMAKE_SOURCE_DIR}/third_party/METIS")

    set(GKLIB_INCLUDE_DIRS "${GKLIB_DIR}/include")
    file(GLOB_RECURSE GKLIB_LIBRARIES "${GKLIB_DIR}/lib/lib*.a")

    set(METIS_INCLUDE_DIRS "${METIS_DIR}/include")
    file(GLOB_RECURSE METIS_LIBRARIES "${METIS_DIR}/lib/lib*.a")

    include_directories(${METIS_INCLUDE_DIRS})

    add_library(metis SHARED "csrc/partition/metis.cpp")
    target_link_libraries(metis PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(metis PRIVATE ${GKLIB_LIBRARIES})
    target_link_libraries(metis PRIVATE ${METIS_LIBRARIES})
endif()

if(WITH_MTMETIS)
    add_definitions(-DWITH_MTMETIS)
    set(MTMETIS_DIR "${CMAKE_SOURCE_DIR}/third_party/mt-metis")

    set(MTMETIS_INCLUDE_DIRS "${MTMETIS_DIR}/include")
    file(GLOB_RECURSE MTMETIS_LIBRARIES "${MTMETIS_DIR}/lib/lib*.a")

    include_directories(${MTMETIS_INCLUDE_DIRS})
    add_library(mtmetis SHARED "csrc/partition/mtmetis.cpp")
    target_link_libraries(mtmetis PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(mtmetis PRIVATE ${MTMETIS_LIBRARIES})

    target_compile_definitions(mtmetis PRIVATE -DMTMETIS_64BIT_VERTICES)
    target_compile_definitions(mtmetis PRIVATE -DMTMETIS_64BIT_EDGES)
    target_compile_definitions(mtmetis PRIVATE -DMTMETIS_64BIT_WEIGHTS)
    target_compile_definitions(mtmetis PRIVATE -DMTMETIS_64BIT_PARTITIONS)
endif()

include_directories("csrc/include")
add_library(${PROJECT_NAME} SHARED csrc/export.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE -DTORCH_EXTENSION_NAME=lib${PROJECT_NAME})

if(WITH_PYTHON)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()

if (WITH_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE uvm)
endif()

if (WITH_METIS)
    target_link_libraries(${PROJECT_NAME} PRIVATE metis)
endif()

if (WITH_MTMETIS)
    target_link_libraries(${PROJECT_NAME} PRIVATE mtmetis)
endif()
