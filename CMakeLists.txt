cmake_minimum_required(VERSION 3.15)

project(starrygl VERSION 0.1)

option(WITH_PYTHON "Link to Python when building" ON)
option(WITH_CUDA "Link to CUDA when building" ON)
option(WITH_METIS "Link to METIS when building" OFF)
option(WITH_MTMETIS "Link to multi-threaded METIS when building" OFF)
option(WITH_LDG "Link to (multi-threaded optionally) LDG when building" OFF)

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

    add_library(uvm_ops SHARED ${UVM_SRCS})
    target_link_libraries(uvm_ops PRIVATE ${TORCH_LIBRARIES})
endif()


if(WITH_METIS)
    # add_definitions(-DWITH_METIS)
    # set(GKLIB_DIR "${CMAKE_SOURCE_DIR}/third_party/GKlib")
    # set(METIS_DIR "${CMAKE_SOURCE_DIR}/third_party/METIS")

    # set(GKLIB_INCLUDE_DIRS "${GKLIB_DIR}/include")
    # file(GLOB_RECURSE GKLIB_LIBRARIES "${GKLIB_DIR}/lib/lib*.a")

    # set(METIS_INCLUDE_DIRS "${METIS_DIR}/include")
    # file(GLOB_RECURSE METIS_LIBRARIES "${METIS_DIR}/lib/lib*.a")

    # include_directories(${METIS_INCLUDE_DIRS})

    # add_library(metis_partition SHARED "csrc/partition/metis.cpp")
    # target_link_libraries(metis_partition PRIVATE ${TORCH_LIBRARIES})
    # target_link_libraries(metis_partition PRIVATE ${GKLIB_LIBRARIES})
    # target_link_libraries(metis_partition PRIVATE ${METIS_LIBRARIES})

    add_definitions(-DWITH_METIS)
    set(METIS_DIR "${CMAKE_SOURCE_DIR}/third_party/METIS")
    set(METIS_GKLIB_DIR "${METIS_DIR}/GKlib")

    file(GLOB METIS_SRCS "${METIS_DIR}/libmetis/*.c")
    file(GLOB METIS_GKLIB_SRCS "${METIS_GKLIB_DIR}/*.c")
    
    if (MSVC)
        file(GLOB METIS_GKLIB_WIN32_SRCS "${METIS_GKLIB_DIR}/win32/*.c")
        set(METIS_GKLIB_SRCS ${METIS_GKLIB_SRCS} ${METIS_GKLIB_WIN32_SRCS})
    endif()

    add_library(metis_partition SHARED
        "csrc/partition/metis.cpp"
        ${METIS_SRCS} ${METIS_GKLIB_SRCS}
    )
    target_include_directories(metis_partition PRIVATE "${METIS_DIR}/include")
    target_include_directories(metis_partition PRIVATE "${METIS_GKLIB_DIR}")
    if (MSVC)
        target_include_directories(metis_partition PRIVATE "${METIS_GKLIB_DIR}/win32")
    endif()

    target_compile_definitions(metis_partition PRIVATE -DIDXTYPEWIDTH=64)
    target_compile_definitions(metis_partition PRIVATE -DREALTYPEWIDTH=32)
    target_compile_options(metis_partition PRIVATE -O3)

    target_link_libraries(metis_partition PRIVATE ${TORCH_LIBRARIES})
    if (UNIX)
        target_link_libraries(metis_partition PRIVATE m)
    endif()
endif()

if(WITH_MTMETIS)
    add_definitions(-DWITH_MTMETIS)
    set(MTMETIS_DIR "${CMAKE_SOURCE_DIR}/third_party/mt-metis")

    set(MTMETIS_INCLUDE_DIRS "${MTMETIS_DIR}/include")
    file(GLOB_RECURSE MTMETIS_LIBRARIES "${MTMETIS_DIR}/lib/lib*.a")

    include_directories(${MTMETIS_INCLUDE_DIRS})
    add_library(mtmetis_partition SHARED "csrc/partition/mtmetis.cpp")
    target_link_libraries(mtmetis_partition PRIVATE ${TORCH_LIBRARIES})
    target_link_libraries(mtmetis_partition PRIVATE ${MTMETIS_LIBRARIES})

    target_compile_definitions(mtmetis_partition PRIVATE -DMTMETIS_64BIT_VERTICES)
    target_compile_definitions(mtmetis_partition PRIVATE -DMTMETIS_64BIT_EDGES)
    target_compile_definitions(mtmetis_partition PRIVATE -DMTMETIS_64BIT_WEIGHTS)
    target_compile_definitions(mtmetis_partition PRIVATE -DMTMETIS_64BIT_PARTITIONS)
endif()

if (WITH_LDG)
    # Imports neighbor-clustering based (e.g. LDG algorithm) graph partitioning implementation
    add_definitions(-DWITH_LDG)
    set(LDG_DIR "third_party/ldg_partition")
    
    add_library(ldg_partition SHARED "csrc/partition/ldg.cpp")
    target_link_libraries(ldg_partition PRIVATE ${TORCH_LIBRARIES})

    add_subdirectory(${LDG_DIR})
    target_include_directories(ldg_partition PRIVATE ${LDG_DIR})
    target_link_libraries(ldg_partition PRIVATE ldg-vertex-partition)
endif ()

include_directories("csrc/include")
add_library(${PROJECT_NAME} SHARED csrc/export.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE -DTORCH_EXTENSION_NAME=lib${PROJECT_NAME})

if(WITH_PYTHON)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()

if (WITH_CUDA)
    target_link_libraries(${PROJECT_NAME} PRIVATE uvm_ops)
endif()

if (WITH_METIS)
    message(STATUS "Current project '${PROJECT_NAME}' uses METIS graph partitioning algorithm.")
    target_link_libraries(${PROJECT_NAME} PRIVATE metis_partition)
endif()

if (WITH_MTMETIS)
    message(STATUS "Current project '${PROJECT_NAME}' uses multi-threaded METIS graph partitioning algorithm.")
    target_link_libraries(${PROJECT_NAME} PRIVATE mtmetis_partition)
endif()

if (WITH_LDG)
    message(STATUS "Current project '${PROJECT_NAME}' uses LDG graph partitioning algorithm.")
    target_link_libraries(${PROJECT_NAME} PRIVATE ldg_partition)
endif()

# add libsampler.so
set(SAMLPER_NAME "${PROJECT_NAME}_sampler")
# set(BOOST_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/third_party/boost_1_83_0")
# include_directories(${BOOST_INCLUDE_DIRS})

file(GLOB_RECURSE SAMPLER_SRCS "csrc/sampler/*.cpp")
add_library(${SAMLPER_NAME} SHARED ${SAMPLER_SRCS})

target_include_directories(${SAMLPER_NAME} PRIVATE "csrc/sampler/include")

target_compile_options(${SAMLPER_NAME} PRIVATE -O3)
message(STATUS "Torch include directories: ${TORCH_LIBRARIES}")
target_link_libraries(${SAMLPER_NAME} PRIVATE ${TORCH_LIBRARIES})
target_compile_definitions(${SAMLPER_NAME} PRIVATE -DTORCH_EXTENSION_NAME=lib${SAMLPER_NAME})

if(WITH_PYTHON)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(${SAMLPER_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()
