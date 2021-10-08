# PROJECT GLOBAL PROPERTIES
if(TPR_PROJECT AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(backward-cpp)
endif()

if(REAL_TYPE STREQUAL "double")
    set_property(
        DIRECTORY ${PROJECT_SOURCE_DIR}/src
        APPEND
        PROPERTY COMPILE_DEFINITIONS _REAL_IS_DOUBLE_)
endif()

if(BUILD_CUDA)
    set_property(
        DIRECTORY ${PROJECT_SOURCE_DIR}/src
        APPEND
        PROPERTY COMPILE_DEFINITIONS BUILD_CUDA)
endif()

set(PROJECT_INC_DIR
    "${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/PMlib/include")
set_property(
    DIRECTORY ${PROJECT_SOURCE_DIR}/src
    APPEND
    PROPERTY COMPILE_DEFINITIONS DISABLE_MPI="ON" COMPILE_DEFINITIONS
             enable_OPENMP="ON")
# END PROJECT GLOBAL PROPERTIES

# CR
add_library(libcr cr.cpp cr.hpp lib.hpp)

# PCR
add_library(libpcr pcr.cpp pcr.hpp lib.hpp)

# TPR_PERF
add_library(tpr_perf tpr_perf.cpp tpr_perf.hpp)
target_include_directories(tpr_perf PRIVATE ${PROJECT_INC_DIR})
target_link_libraries(tpr_perf PRIVATE pmlib)
if(TPR_PERF)
    set_property(
        TARGET tpr_perf
        APPEND
        PROPERTY COMPILE_DEFINITIONS TPR_PERF="yes")
endif()

# TPR
add_library(tpr tpr.cpp)
target_include_directories(tpr PUBLIC ${PROJECT_INC_DIR})
target_link_libraries(tpr PUBLIC pmlib libcr tpr_perf)

# PTPR
add_library(ptpr ptpr.cpp)
target_include_directories(ptpr PUBLIC ${PROJECT_INC_DIR})
target_link_libraries(ptpr PUBLIC pmlib libpcr tpr_perf)

if(${OPT_PAPI})
    target_link_libraries(ptpr PUBLIC papi_ext "${PAPI_STATIC_LIB}")
    target_include_directories(ptpr PUBLIC ${PAPI_DIR}/include)
    target_link_directories(ptpr PUBLIC ${PAPI_DIR}/lib)
endif()

if(BUILD_CUDA)
    add_library(tpr_cu STATIC tpr.cu tpr.cuh)
    set_target_properties(tpr_cu PROPERTIES POSITION_INDEPENDENT_CODE ON
                                            CUDA_SEPARABLE_COMPILATION ON)
    add_library(ptpr_cu STATIC ptpr.cu ptpr.cuh)
    set_target_properties(ptpr_cu PROPERTIES POSITION_INDEPENDENT_CODE ON
                                             CUDA_SEPARABLE_COMPILATION ON)
    add_library(ref_cusp STATIC reference_cusparse.cu reference_cusparse.cuh)
    set_target_properties(ref_cusp PROPERTIES POSITION_INDEPENDENT_CODE ON
                                              CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(ref_cusp PRIVATE cusparse cublas cudart)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        set_property(
            TARGET tpr_cu
            APPEND
            PROPERTY CUDA_ARCHITECTURES ${CU_ARCH})
        set_property(
            TARGET ptpr_cu
            APPEND
            PROPERTY CUDA_ARCHITECTURES ${CU_ARCH})
        set_property(
            TARGET ref_cusp
            APPEND
            PROPERTY CUDA_ARCHITECTURES ${CU_ARCH})
    endif()

    target_include_directories(tpr_cu PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(ptpr_cu PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_include_directories(ref_cusp PUBLIC ${PROJECT_SOURCE_DIR}/src)

    add_executable(cumain main.cu)
    target_include_directories(cumain PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(cumain PRIVATE trisys tpr_cu ptpr_cu ref_cusp)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        set_property(
            TARGET cumain
            APPEND
            PROPERTY CUDA_ARCHITECTURES ${CU_ARCH})
    endif()

    # FOR BENCHMARK PROGRAM
    add_executable(
        cupm
        pm.cu
        pm.cuh
        tpr.cu
        tpr.cuh
        ptpr.cu
        ptpr.cuh
        reference_cusparse.cu
        reference_cusparse.cuh)
    target_link_libraries(cupm PUBLIC cusparse cublas cudart trisys)
    set_target_properties(
        cupm
        PROPERTIES COMPILE_DEFINITIONS TPR_PERF="ON"
                   POSITION_INDEPENDENT_CODE ON
                   CUDA_SEPARABLE_COMPILATION ON)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
        set_property(
            TARGET cupm
            APPEND
            PROPERTY CUDA_ARCHITECTURES ${CU_ARCH})
    endif()
endif()

# build sample main and benchmark program
if(TPR_PROJECT)
    # Tridiagonal system
    add_library(trisys system.cpp system.hpp lib.hpp)
    target_link_libraries(trisys PRIVATE effolkronium_random)

    # tpr_main
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_executable(tpr_main main.cpp ${BACKWARD_ENABLE})
        target_include_directories(
            tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/src/backward-cpp)
        add_backward(tpr_main)
    else() # Release etc.
        add_executable(tpr_main main.cpp)
    endif()
    target_link_libraries(tpr_main PUBLIC trisys libcr libpcr tpr ptpr pmlib)
    target_include_directories(tpr_main PUBLIC ${PROJECT_INC_DIR})
    set_property(TARGET tpr_main PROPERTY COMPILE_DEFINITIONS TPR_PERF="off")

    if(BUILD_CUDA)
        target_link_libraries(tpr_main PRIVATE tpr_cu ptpr_cu)
        target_include_directories(tpr_main PRIVATE ${CUDA_INCLUDE_DIRS})

        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
            set_property(
                TARGET tpr_main
                APPEND
                PROPERTY CUDA_ARCHITECTURES ${CU_ARCH})
        endif()
    endif()

    # PMlib
    set(pmlib_srcdir ${PROJECT_SOURCE_DIR}/PMlib/src)
    set(pmlib_incdir ${PROJECT_SOURCE_DIR}/PMlib/include)
    set(pm_files
        ${pmlib_srcdir}/PerfCpuType.cpp ${pmlib_srcdir}/PerfMonitor.cpp
        ${pmlib_srcdir}/PerfWatch.cpp ${pmlib_srcdir}/PerfProgFortran.cpp)
    configure_file(${pmlib_incdir}/pmVersion.h.in ${pmlib_incdir}/pmVersion.h
                   @ONLY)

    add_library(pmlib STATIC ${pm_files})
    target_include_directories(pmlib PUBLIC ${pmlib_incdir})
    set_property(TARGET pmlib PROPERTY COMPILE_DEFINITIONS DISABLE_MPI="ON"
                                       COMPILE_DEFINITIONS enable_OPENMP="ON")

    if(with_PAPI) # with_PAPI=[path/to/papi]
        message("PAPI DIR: ${PAPI_DIR}")
        set_property(
            TARGET pmlib
            APPEND
            PROPERTY COMPILE_DEFINITIONS with_PAPI="${with_PAPI}")
        set_property(
            TARGET pmlib
            APPEND
            PROPERTY COMPILE_DEFINITIONS PAPI_DIR="${PAPI_DIR}")

        # link static library
        set(PAPI_STATIC_LIB ${PAPI_DIR}/lib/libpapi.a ${PAPI_DIR}/lib/libpfm.a)

        set(PAPI_EXT_DIR "${PROJECT_SOURCE_DIR}/PMlib/src_papi_ext")
        add_library(papi_ext STATIC ${PAPI_EXT_DIR}/papi_ext.c)

        set(CMAKE_C_FLAGS
            "${CMAKE_C_FLAGS} -DSTATIC_PAPI_EVENTS_TABLE -D_REENTRANT -D_GNU_SOURCE"
        )
        set(CMAKE_C_FLAGS
            "${CMAKE_C_FLAGS} -DUSE_COMPILER_TLS  -DPAPI_NO_MEMORY_MANAGEMENT -DUSE_PAPI"
        )
        target_include_directories(papi_ext PUBLIC ${PAPI_DIR}/include
                                                   ${PAPI_DIR}/lib)
        set_target_properties(papi_ext PROPERTIES LINKER_LANGUAGE CXX)
        target_link_libraries(papi_ext ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES})

        # link papi to PMlib
        target_link_libraries(pmlib papi_ext)
        target_include_directories(pmlib PUBLIC ${PAPI_DIR}/include)
        target_link_directories(pmlib PUBLIC ${PAPI_DIR}/lib)
    endif()

    # FOR BENCHMARK PROGRAM
    add_executable(tpr_pm pm.cpp)
    target_link_libraries(tpr_pm PUBLIC trisys tpr ptpr libpcr pmlib
                                        effolkronium_random)
    if(${OPT_PAPI})
        target_link_libraries(tpr_pm PUBLIC papi_ext "${PAPI_STATIC_LIB}")
        target_include_directories(tpr_pm PUBLIC ${PAPI_DIR}/include)
        target_link_directories(tpr_pm PUBLIC ${PAPI_DIR}/lib)
    endif()
    target_include_directories(
        tpr_pm PUBLIC ${PROJECT_SOURCE_DIR}/src
                      ${PROJECT_SOURCE_DIR}/PMlib/include)
    set_property(TARGET tpr_pm PROPERTY COMPILE_DEFINITIONS DISABLE_MPI="ON")
endif()
