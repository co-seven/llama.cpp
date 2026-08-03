set(TARGET_NAME ggml-spacemit)

ggml_add_backend_library(${TARGET_NAME}
    ggml-spacemit.cpp
    ime.cpp
    ime1_kernels.cpp
    ime2_kernels.cpp
    ime_env.cpp
    rvv_kernels.cpp
    repack.cpp
    spine_mem_pool.cpp
    ../../include/ggml-spacemit.h)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cpu)

target_compile_definitions(${TARGET_NAME} PRIVATE
    GGML_USE_CPU_RISCV64_SPACEMIT
)

if(RISCV_ROOT_PATH)
    target_compile_definitions(${TARGET_NAME} PRIVATE GGML_CPU_RISCV64_SPACEMIT)
endif()
