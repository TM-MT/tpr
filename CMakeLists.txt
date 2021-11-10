cmake_minimum_required(VERSION 3.0)

set(TPR_PROJECT ON)
if(DEFINED PROJECT_NAME)
    set(TPR_PROJECT OFF)
endif()

option(BUILD_CUDA "Build {tpr|ptpr}.cu" off)
# require this from cmake 3.18
option(CU_ARCH "CUDA ARCHITECTURE" "")
option(ENABLE_TEST "MAKE TEST" off)
option(TPR_PERF "Enable PMlib in TPR" "no")
option(with_PAPI "Enable PAPI" "OFF")
option(with_LAPACK "Enable Reference LAPACK" "no")

if(BUILD_CUDA)
    project(tpr C CXX CUDA)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        include(FindCUDAToolkit 11.0)
    else()
        find_package(CUDA 11.0 REQUIRED)
    endif()
else()
    project(tpr C CXX)
endif()

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(ENABLE_TEST)
    enable_testing()
    add_subdirectory(googletest)
    add_subdirectory(test)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 17)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # increase alias check param
        set(CMAKE_CXX_FLAGS
            "${CMAKE_CXX_FLAGS} --param vect-max-version-for-alias-checks=50")
    endif()
endif()

if(with_LAPACK)
    set(BLA_SIZEOF_INTEGER 8)
    include(FindLAPACK)
endif()

if(with_PAPI)
    add_definitions(-DUSE_PAPI)
    set(OPT_PAPI "ON")
    set(PAPI_DIR "${with_PAPI}")
endif()

if(TPR_PROJECT)
    add_subdirectory(${PROJECT_SOURCE_DIR}/random)
    set(Random_BuildTests "no")
endif()
add_subdirectory(src)
