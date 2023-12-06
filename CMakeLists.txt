cmake_minimum_required(VERSION 3.15)

project(starrygl_ops VERSION 0.1)

option(WITH_PYTHON "Link to Python when building" ON)

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

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenMP REQUIRED)

# add_subdirectory(third_party/cccl-2.2.0)
# add_subdirectory(third_party/cutlass-3.2.1)

# add_library(${PROJECT_NAME} SHARED csrc/custom_ops.cpp csrc/custom_kernel.cu)
add_library(${PROJECT_NAME} SHARED csrc/custom_ops.cpp)

if(WITH_PYTHON)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_PYTHON_LIBRARY})
endif()
target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
# target_link_libraries(${PROJECT_NAME} PRIVATE CCCL::CCCL)

target_compile_definitions(${PROJECT_NAME} PRIVATE -DTORCH_EXTENSION_NAME=lib${PROJECT_NAME})

# set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" OUTPUT_NAME "_C")
install(TARGETS ${PROJECT_NAME} DESTINATION  "${CMAKE_SOURCE_DIR}/starrygl/lib")
