
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


# PMlib
set(pmlib_srcdir ${PROJECT_SOURCE_DIR}/PMlib/src)
set(pmlib_incdir ${PROJECT_SOURCE_DIR}/PMlib/include)
set(pm_files
       ${pmlib_srcdir}/PerfCpuType.cpp
       ${pmlib_srcdir}/PerfMonitor.cpp
       ${pmlib_srcdir}/PerfWatch.cpp
       ${pmlib_srcdir}/PerfProgFortran.cpp)
configure_file( ${pmlib_incdir}/pmVersion.h.in ${pmlib_incdir}/pmVersion.h @ONLY)

add_library(pmlib STATIC ${pm_files})
target_include_directories(pmlib PUBLIC ${pmlib_incdir})
set_property(TARGET pmlib PROPERTY COMPILE_DEFINITIONS DISABLE_MPI="ON" )


# FOR BENCHMARK PROGRAM
add_executable(tpr_pm
    pm.cpp pcr.cpp lib.cpp tpr.cpp
)
target_link_libraries(tpr_pm pmlib)
target_include_directories(tpr_pm PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(tpr_pm PUBLIC ${PROJECT_SOURCE_DIR}/dbg-macro)
target_include_directories(tpr_pm PUBLIC ${PROJECT_SOURCE_DIR}/PMlib/include)
set_property(TARGET tpr_pm PROPERTY COMPILE_DEFINITIONS DISABLE_MPI="ON" )
target_link_libraries(tpr_pm effolkronium_random) 
