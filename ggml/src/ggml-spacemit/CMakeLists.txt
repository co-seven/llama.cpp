set(TARGET_NAME ggml-spacemit)

# spine-runtime is a hard dependency (submodule at thirdparty/spine-runtime)
get_filename_component(SPINE_RUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/spine-runtime" ABSOLUTE)
if(NOT EXISTS "${SPINE_RUNTIME_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "spine-runtime submodule not found at ${SPINE_RUNTIME_DIR}. Run: git submodule update --init --recursive")
endif()

# Build spine-runtime as a static library and link it in
set(SPINE_RUNTIME_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPINE_RUNTIME_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${SPINE_RUNTIME_DIR} ${CMAKE_BINARY_DIR}/spine-runtime-build)

# ggml-spacemit and ggml-cpu/spacemit are mutually exclusive.
# When GGML_SPACEMIT is ON, force GGML_CPU_RISCV64_SPACEMIT OFF.
if(GGML_SPACEMIT AND GGML_CPU_RISCV64_SPACEMIT)
    message(STATUS "GGML_SPACEMIT is ON -> disabling GGML_CPU_RISCV64_SPACEMIT")
    set(GGML_CPU_RISCV64_SPACEMIT OFF CACHE BOOL "" FORCE)
endif()

# Detect IME support (inline version of FindSMTIME.cmake)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    include(CheckCSourceCompiles)
    set(SMT_MARCH_STR "-march=rv64gcv_zfh_zvfh_zba_zicbop")
    if((CMAKE_C_COMPILER_ID STREQUAL "GNU" AND
        CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 15) OR
       CMAKE_C_COMPILER_ID MATCHES "Clang")
        string(APPEND SMT_MARCH_STR "_xsmtvdotii")
    endif()
    set(CMAKE_REQUIRED_FLAGS "${SMT_MARCH_STR}")

    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot v2, v0, v1\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_IME1)
    check_c_source_compiles("int main() {__asm__ volatile(\"vmadot v2, v0, v1, i4\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S4)
    check_c_source_compiles("int main() {__asm__ volatile(\"vpack.vv v2, v0, v1, 2\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VPACK)
    check_c_source_compiles("int main() {__asm__ volatile(\"vnspack.vv v2, v0, v1, 2\");}" SPACEMIT_RISCV_COMPILER_SUPPORT_VNPACK)
    unset(CMAKE_REQUIRED_FLAGS)

    set(RISCV64_SPACEMIT_IME_SPEC "")
    if(SPACEMIT_RISCV_COMPILER_SUPPORT_IME1)
        set(RISCV64_SPACEMIT_IME_SPEC "RISCV64_SPACEMIT_IME1")
    endif()
    if(SPACEMIT_RISCV_COMPILER_SUPPORT_VMADOT_S4 AND SPACEMIT_RISCV_COMPILER_SUPPORT_VPACK AND SPACEMIT_RISCV_COMPILER_SUPPORT_VNPACK)
        list(APPEND RISCV64_SPACEMIT_IME_SPEC "RISCV64_SPACEMIT_IME2")
    endif()
    message("RISCV64_SPACEMIT_IME_SPEC: ${RISCV64_SPACEMIT_IME_SPEC}")
endif()

# NOTE: ime_env.cpp is intentionally excluded. global_spine_env_info is
# provided by ggml-spacemit.cpp, initialized from spert::backend_info()
# and GGML_SPACEMIT_WORKERS env var.
ggml_add_backend_library(${TARGET_NAME}
    ggml-spacemit.cpp
    ime.cpp
    ime1_kernels.cpp
    ime2_kernels.cpp
    rvv_kernels.cpp
    repack.cpp
    spine_mem_pool.cpp
    ../../include/ggml-spacemit.h)

target_link_libraries(${TARGET_NAME} PRIVATE spert)

target_include_directories(${TARGET_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ggml-cpu
    ${SPINE_RUNTIME_DIR}/include)

target_compile_definitions(${TARGET_NAME} PRIVATE
    GGML_USE_CPU_RISCV64_SPACEMIT
    ${RISCV64_SPACEMIT_IME_SPEC}
)

# Pass the full march string to both targets
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    set(SPACEMIT_MARCH "-march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicbop_zihintpause_zawrs_zicond_zvbb")
    if(RISCV64_SPACEMIT_IME_SPEC MATCHES "IME")
        set(SPACEMIT_MARCH "-march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicbop_zihintpause_zawrs_zicond_zvbb_xsmtvdotii")
    endif()
    target_compile_options(spert PRIVATE -march=rv64gcv_zfh_zvfh_zba_zbb_zbs_zicbop_zihintpause_zawrs_zicond_zvbb)
    target_compile_options(${TARGET_NAME} PRIVATE ${SPACEMIT_MARCH})
endif()
