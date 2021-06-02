
message("${LIBBFD_LIBRARY}")

add_subdirectory(backward-cpp)


add_executable(tpr_main 
    main.cpp pcr.cpp lib.cpp tpr.cpp 
    ${BACKWARD_ENABLE})
set_target_properties(tpr_main PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -lbfd -ldl ")
# target_link_libraries(tpr_main "${LIBBFD_LIBRARY}")
# target_include_directories(tpr_main PUBLIC "${LIBBFD_INCLUDE_DIR}")

target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/dbg-macro)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/src/backward-cpp)
    add_backward(tpr_main)
endif()

