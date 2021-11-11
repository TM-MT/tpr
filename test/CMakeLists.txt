set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(GTEST_INC "${PROJECT_SOURCE_DIR}/googletest/googletest/include")

add_executable(examples examples.cpp)
target_include_directories(examples PUBLIC ${SRC_DIR}
                                           ${PROJECT_SOURCE_DIR}/PMlib/include)
target_link_libraries(
    examples
    libcr
    libpcr
    tpr
    ptpr
    trisys
    gtest_main)
set_property(TARGET examples PROPERTY COMPILE_DEFINITIONS DISABLE_MPI="ON"
                                      COMPILE_DEFINITIONS enable_OPENMP="ON")

if(BUILD_CUDA)
    add_executable(cuexamples cuexamples.cpp)
    target_include_directories(
        cuexamples PUBLIC ${SRC_DIR} ${PROJECT_SOURCE_DIR}/PMlib/include)
    target_link_libraries(cuexamples tpr_cu ptpr_cu trisys gtest_main)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        set_property(
            TARGET cuexamples
            APPEND
            PROPERTY CUDA_ARCHITECTURES ${CU_ARCH})
    endif()
endif()

include(GoogleTest)
gtest_discover_tests(examples)
if(BUILD_CUDA)
    gtest_discover_tests(cuexamples)
endif()
