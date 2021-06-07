cmake_minimum_required(VERSION 3.0)

project(tpr C CXX)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DEBUG_POSTFIX d)


if(with_PAPI)
    add_definitions(-DUSE_PAPI)
    set(OPT_PAPI "ON")
    set(PAPI_DIR "${with_PAPI}")
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/structopt)
add_subdirectory(${PROJECT_SOURCE_DIR}/random)
add_subdirectory(src)
