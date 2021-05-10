
message("${LIBBFD_LIBRARY}")

add_subdirectory(backward-cpp)

add_executable(tpr_main 
    main.cpp pcr.cpp lib.cpp tpr.cpp 
    ${BACKWARD_ENABLE})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -lbfd -ldl ")
# target_link_libraries(tpr_main "${LIBBFD_LIBRARY}")
# target_include_directories(tpr_main PUBLIC "${LIBBFD_INCLUDE_DIR}")

target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/dbg-macro)
target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/src/backward-cpp)
add_backward(tpr_main)

