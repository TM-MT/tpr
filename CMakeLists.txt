cmake_minimum_required(VERSION 3.0)

set(TPR_PROJECT ON)
if (DEFINED PROJECT_NAME)
    set(TPR_PROJECT OFF)
endif()


option(BUILD_CUDA "Build {tpr|ptpr}.cu" off)
# require this from cmake 3.18
option(CU_ARCH "CUDA ARCHITECTURE" "")
option(TPR_PERF "Enable PMlib in TPR" "no")
option (with_PAPI "Enable PAPI" "OFF")

if (BUILD_CUDA)
    project(tpr C CXX CUDA)
else()
    project(tpr C CXX)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DEBUG_POSTFIX d)


if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # increase alias check param
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --param vect-max-version-for-alias-checks=50")
    endif()
endif()


if(with_PAPI)
    add_definitions(-DUSE_PAPI)
    set(OPT_PAPI "ON")
    set(PAPI_DIR "${with_PAPI}")
endif()

if (TPR_PROJECT)
    add_subdirectory(${PROJECT_SOURCE_DIR}/random)
    set(Random_BuildTests "no")
endif()
add_subdirectory(src)
