
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(backward-cpp)
endif()

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# PCR
add_library(libpcr
    pcr.cpp lib.cpp)

target_include_directories(libpcr PUBLIC ${PROJECT_SOURCE_DIR}/src)


# TPR
add_library(libtpr
    tpr.cpp lib.cpp)

target_include_directories(libtpr PUBLIC ${PROJECT_SOURCE_DIR}/src)

# build sample main and benchmark program
if (TPR_PROJECT)
    # tpr_main
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_executable(tpr_main
            main.cpp
            ${BACKWARD_ENABLE}
        )
        set_target_properties(tpr_main PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
        target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/src/backward-cpp)
        add_backward(tpr_main)
    else() # Release etc.
        add_executable(tpr_main
            main.cpp
        )
    endif()
    target_link_libraries(tpr_main PUBLIC libpcr)
    target_link_libraries(tpr_main PUBLIC libtpr)
    target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/dbg-macro)


    # PMlib
    set(pmlib_srcdir ${PROJECT_SOURCE_DIR}/PMlib/src)
    set(pmlib_incdir ${PROJECT_SOURCE_DIR}/PMlib/include)
    set(pm_files
           ${pmlib_srcdir}/PerfCpuType.cpp
           ${pmlib_srcdir}/PerfMonitor.cpp
           ${pmlib_srcdir}/PerfWatch.cpp
           ${pmlib_srcdir}/PerfProgFortran.cpp)
    configure_file( ${pmlib_incdir}/pmVersion.h.in ${pmlib_incdir}/pmVersion.h @ONLY)

    option (with_PAPI "Enable PAPI" "OFF")

    add_library(pmlib STATIC ${pm_files})
    target_include_directories(pmlib PUBLIC ${pmlib_incdir})
    set_property(TARGET pmlib PROPERTY COMPILE_DEFINITIONS DISABLE_MPI="ON" )
    set_property(TARGET pmlib APPEND PROPERTY COMPILE_DEFINITIONS enable_OPENMP="ON" )

    if (with_PAPI) # with_PAPI=[path/to/papi]
        message("PAPI DIR: ${PAPI_DIR}")
        set_property(TARGET pmlib APPEND PROPERTY COMPILE_DEFINITIONS with_PAPI="${with_PAPI}" )
        set_property(TARGET pmlib APPEND PROPERTY COMPILE_DEFINITIONS PAPI_DIR="${PAPI_DIR}" )

        # link static library
        set(PAPI_STATIC_LIB ${PAPI_DIR}/lib/libpapi.a
          ${PAPI_DIR}/lib/libpfm.a
        )

        set(PAPI_EXT_DIR "${PROJECT_SOURCE_DIR}/PMlib/src_papi_ext")
        add_library(papi_ext STATIC ${PAPI_EXT_DIR}/papi_ext.c)

        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DSTATIC_PAPI_EVENTS_TABLE -D_REENTRANT -D_GNU_SOURCE")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_COMPILER_TLS  -DPAPI_NO_MEMORY_MANAGEMENT -DUSE_PAPI")
        target_include_directories(papi_ext PUBLIC ${PAPI_DIR}/include)
        target_link_directories(papi_ext PUBLIC ${PAPI_DIR}/lib)
        set_target_properties(papi_ext PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(papi_ext ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})

        target_link_libraries(pmlib papi_ext)
        target_include_directories(pmlib PUBLIC ${PAPI_DIR}/include)
        target_link_directories(pmlib PUBLIC ${PAPI_DIR}/lib)
    endif()

    # FOR BENCHMARK PROGRAM
    add_executable(tpr_pm
        pm.cpp
    )
    target_link_libraries(tpr_pm libpcr)
    target_link_libraries(tpr_pm libtpr)
    target_link_libraries(tpr_pm pmlib)
    target_link_libraries(tpr_pm structopt)
    if (${OPT_PAPI})
        target_link_libraries(tpr_pm papi_ext)
        target_include_directories(tpr_pm PUBLIC ${PAPI_DIR}/include)
        target_link_directories(tpr_pm PUBLIC ${PAPI_DIR}/lib)
        target_link_libraries(tpr_pm "${PAPI_STATIC_LIB}") 
    endif()
    target_include_directories(tpr_pm PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(tpr_pm PUBLIC ${PROJECT_SOURCE_DIR}/dbg-macro)
    target_include_directories(tpr_pm PUBLIC ${PROJECT_SOURCE_DIR}/PMlib/include)
    set_property(TARGET tpr_pm PROPERTY COMPILE_DEFINITIONS DISABLE_MPI="ON" )
    target_link_libraries(tpr_pm effolkronium_random)
endif()

if (REAL_TYPE STREQUAL "double")
    set_property(DIRECTORY ${PROJECT_SOURCE_DIR}/src
                APPEND PROPERTY COMPILE_DEFINITIONS _REAL_IS_DOUBLE_)
endif()
