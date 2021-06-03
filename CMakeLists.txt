cmake_minimum_required(VERSION 3.0)

project(tpr CXX)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_DEBUG_POSTFIX d)


add_subdirectory(${PROJECT_SOURCE_DIR}/random)
add_subdirectory(src)
