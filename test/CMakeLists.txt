
set(SRC_DIR "${PROJECT_SOURCE_DIR}/src")
set(GTEST_INC "${PROJECT_SOURCE_DIR}/googletest/googletest/include")


add_executable(examples examples.cpp)
target_include_directories(examples PUBLIC ${SRC_DIR})
target_link_libraries(examples libcr libpcr tpr ptpr gtest_main)

# add_test(NAME examples COMMAND examples)

include(GoogleTest)
gtest_discover_tests(examples)
