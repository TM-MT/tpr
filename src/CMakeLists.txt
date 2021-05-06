

add_executable(tpr_main main.cpp pcr.cpp lib.cpp)

target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(tpr_main PUBLIC ${PROJECT_SOURCE_DIR}/dbg-macro)

