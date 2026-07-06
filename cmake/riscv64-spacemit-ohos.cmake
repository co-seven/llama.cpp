# Toolchain file for cross-compiling llama.cpp for RISC-V 64 OpenHarmony (OHOS)
# using the SpacemiT musl clang toolchain
# (spacemit-toolchain-linux-musl-x86_64-oh-*).
#
# Mirrors spacemit-ep/cmake/oh_riscv64.toolchain.cmake, but leaves the
# `-march`/`-mabi` selection to ggml's own CPU feature detection
# (ggml/src/ggml-cpu/CMakeLists.txt), so we only inject the OHOS-specific
# global flags here (musl libc++ static linking, __OHOS__, stack size, lld).
#
# Usage:
#   export RISCV_ROOT_PATH=/path/to/spacemit-toolchain-linux-musl-x86_64-oh-*
#   cmake -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-spacemit-ohos.cmake ...

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_SYSTEM_VERSION 1)

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(riscv)")
    message(STATUS "HOST SYSTEM ${CMAKE_HOST_SYSTEM_PROCESSOR}")
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_ASM_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
else()
    if(DEFINED ENV{RISCV_ROOT_PATH})
        file(TO_CMAKE_PATH $ENV{RISCV_ROOT_PATH} RISCV_ROOT_PATH)
    else()
        message(FATAL_ERROR "RISCV_ROOT_PATH env must be defined")
    endif()

    set(RISCV_ROOT_PATH ${RISCV_ROOT_PATH}
        CACHE STRING "root path to riscv ohos toolchain")
    set(CMAKE_C_COMPILER "${RISCV_ROOT_PATH}/bin/clang")
    set(CMAKE_ASM_COMPILER "${RISCV_ROOT_PATH}/bin/clang")
    set(CMAKE_CXX_COMPILER "${RISCV_ROOT_PATH}/bin/clang++")
    set(CMAKE_STRIP ${RISCV_ROOT_PATH}/bin/llvm-strip)
    set(CMAKE_FIND_ROOT_PATH ${RISCV_ROOT_PATH})
    set(CMAKE_SYSROOT "${RISCV_ROOT_PATH}/sysroot")
    set(CMAKE_INCLUDE_PATH "${RISCV_ROOT_PATH}/sysroot/usr/include/")
    set(CMAKE_LIBRARY_PATH "${RISCV_ROOT_PATH}/sysroot/usr/lib/")
    set(CMAKE_PROGRAM_PATH "${RISCV_ROOT_PATH}/sysroot/usr/bin/")
    set(CMAKE_CROSSCOMPILING TRUE)
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

add_definitions(-D__OHOS__)

set(STACK_SIZE_BYTES 16777216)

# NOTE: -march / -mabi are intentionally NOT set here. ggml appends the
# correct `-march=rv64gc..._zfh_zvfh...` and `-mabi=lp64d` itself based on the
# GGML_RV_* options passed on the cmake command line. Setting them here as well
# would duplicate/conflict with ggml's selection.

set(CMAKE_C_FLAGS
    "-Wno-unused-command-line-argument -fuse-ld=lld -Wl,-z,stack-size=${STACK_SIZE_BYTES} ${CMAKE_C_FLAGS}"
)
set(CMAKE_CXX_FLAGS
    "-Wno-unused-command-line-argument -fuse-ld=lld -stdlib=libc++ -static-libstdc++ -Wl,--push-state,-Bstatic -lc++ -lc++abi -Wl,--pop-state -Wl,-z,stack-size=${STACK_SIZE_BYTES} ${CMAKE_CXX_FLAGS}"
)

# NOTE: -Wl,--strip-debug drops only the .debug* DWARF sections at link time.
# The SpacemiT musl toolchain's static libc++/libc++abi/libgcc archives carry
# debug info, and static-linking them (-static-libstdc++ -Bstatic -lc++ ...)
# would otherwise bloat every .so/exe by several MB of DWARF we never ship.
# --strip-debug keeps the .symtab symbol table (function names for backtraces)
# and does not affect runtime behaviour; it only removes debugger metadata.
set(CMAKE_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -static-libgcc -static-libstdc++ -Wl,--push-state,-Bstatic -lgcc -lc++ -lc++abi -Wl,--pop-state -lm -Wl,--strip-debug -Wl,-z,stack-size=${STACK_SIZE_BYTES}"
)

set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} -latomic -lm -Wl,--strip-debug -Wl,-z,stack-size=${STACK_SIZE_BYTES}"
)
set(CMAKE_MODULE_LINKER_FLAGS
    "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--strip-debug -Wl,-z,stack-size=${STACK_SIZE_BYTES}")
