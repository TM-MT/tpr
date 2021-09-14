cmake_minimum_required(VERSION 3.0)

set(TPR_PROJECT ON)
if (DEFINED PROJECT_NAME)
    set(TPR_PROJECT OFF)
endif()

project(tpr C CXX CUDA)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DEBUG_POSTFIX d)


if(with_PAPI)
    add_definitions(-DUSE_PAPI)
    set(OPT_PAPI "ON")
    set(PAPI_DIR "${with_PAPI}")
endif()

if (TPR_PROJECT)
    add_subdirectory(${PROJECT_SOURCE_DIR}/structopt)
    add_subdirectory(${PROJECT_SOURCE_DIR}/random)
    set(Random_BuildTests "no")
endif()
add_subdirectory(src)
