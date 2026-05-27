cmake_minimum_required(VERSION 3.25)

# CLion can prepend its bundled MinGW tools to PATH even when CMAKE_CXX_COMPILER
# points at Rtools. Tell GCC where its matching as.exe/ld.exe live before CMake
# performs compiler checks.
set(LAVAAN_MINGW_BIN "" CACHE PATH "Directory containing the MinGW/Rtools gcc, g++, as, and ld tools")

if(WIN32 AND NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    set(_lavaan_mingw_bin "${LAVAAN_MINGW_BIN}")

    if(NOT _lavaan_mingw_bin AND DEFINED CMAKE_CXX_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL "")
        get_filename_component(_lavaan_mingw_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    endif()

    if(NOT _lavaan_mingw_bin AND DEFINED CMAKE_C_COMPILER AND NOT CMAKE_C_COMPILER STREQUAL "")
        get_filename_component(_lavaan_mingw_bin "${CMAKE_C_COMPILER}" DIRECTORY)
    endif()

    foreach(_lavaan_fallback_bin
            "C:/rtools45/x86_64-w64-mingw32.static.posix/bin"
            "C:/msys64/mingw64/bin")
        if(NOT _lavaan_mingw_bin AND EXISTS "${_lavaan_fallback_bin}/g++.exe")
            set(_lavaan_mingw_bin "${_lavaan_fallback_bin}")
        endif()
    endforeach()

    if(_lavaan_mingw_bin
            AND EXISTS "${_lavaan_mingw_bin}/g++.exe"
            AND EXISTS "${_lavaan_mingw_bin}/as.exe"
            AND EXISTS "${_lavaan_mingw_bin}/ld.exe")
        file(TO_CMAKE_PATH "${_lavaan_mingw_bin}" _lavaan_mingw_bin)
        file(TO_NATIVE_PATH "${_lavaan_mingw_bin}" _lavaan_mingw_bin_native)

        set(ENV{PATH} "${_lavaan_mingw_bin_native};$ENV{PATH}")

        if(NOT DEFINED CMAKE_C_COMPILER OR CMAKE_C_COMPILER STREQUAL "")
            set(CMAKE_C_COMPILER "${_lavaan_mingw_bin}/gcc.exe" CACHE FILEPATH "C compiler")
        endif()

        if(NOT DEFINED CMAKE_CXX_COMPILER OR CMAKE_CXX_COMPILER STREQUAL "")
            set(CMAKE_CXX_COMPILER "${_lavaan_mingw_bin}/g++.exe" CACHE FILEPATH "C++ compiler")
        endif()

        if(EXISTS "${_lavaan_mingw_bin}/windres.exe")
            set(CMAKE_RC_COMPILER "${_lavaan_mingw_bin}/windres.exe" CACHE FILEPATH "RC compiler")
        endif()

        set(_lavaan_binutils_prefix "-B${_lavaan_mingw_bin}/")

        foreach(_lavaan_flag_var
                CMAKE_C_FLAGS_INIT
                CMAKE_CXX_FLAGS_INIT
                CMAKE_EXE_LINKER_FLAGS_INIT
                CMAKE_SHARED_LINKER_FLAGS_INIT
                CMAKE_MODULE_LINKER_FLAGS_INIT)
            string(FIND "${${_lavaan_flag_var}}" "${_lavaan_binutils_prefix}" _lavaan_prefix_pos)
            if(_lavaan_prefix_pos EQUAL -1)
                string(PREPEND ${_lavaan_flag_var} "${_lavaan_binutils_prefix} ")
            endif()
        endforeach()
    endif()
endif()

project(lavaan VERSION 0.1.0 LANGUAGES CXX)

# ============================================================================
# Compiler Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(DEFINED _lavaan_binutils_prefix)
    string(FIND "${CMAKE_CXX_FLAGS}" "${_lavaan_binutils_prefix}" _lavaan_cxx_prefix_pos)
    if(_lavaan_cxx_prefix_pos EQUAL -1)
        add_compile_options("${_lavaan_binutils_prefix}")
    endif()

    string(FIND "${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS}"
            "${_lavaan_binutils_prefix}" _lavaan_link_prefix_pos)
    if(_lavaan_link_prefix_pos EQUAL -1)
        add_link_options("${_lavaan_binutils_prefix}")
    endif()
endif()

# vcpkg triplet (set before find_package calls if not using toolchain)
set(VCPKG_TARGET_TRIPLET "x64-mingw-dynamic" CACHE STRING "vcpkg triplet")

# ============================================================================
# Build Options
# ============================================================================
set(_lavaan_has_tests OFF)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    set(_lavaan_has_tests ON)
endif()

option(lavaan_BUILD_TESTS "Build unit tests" ${_lavaan_has_tests})
option(lavaan_BUILD_BENCHMARKS "Build benchmarks" ON)
option(lavaan_ENABLE_OPENMP "Enable OpenMP for parallelization" ON)
option(lavaan_ENABLE_AVX2 "Enable AVX2 SIMD instructions" ON)

# ============================================================================
# Compiler Flags
# ============================================================================
set(LAVAAN_COMPILE_OPTIONS)
set(LAVAAN_COMPILE_DEFINITIONS)

if(MSVC)
    list(APPEND LAVAAN_COMPILE_OPTIONS /W4 /permissive-)
else()
    list(APPEND LAVAAN_COMPILE_OPTIONS
            -Wall
            -Wextra
            -Wpedantic
            -Wconversion
            -Wsign-conversion
    )
endif()

# Release optimizations
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
endif()

# SIMD flags (Intel Ultra 7 supports AVX2 and AVX-512, but AVX2 is more portable)
if(lavaan_ENABLE_AVX2)
    if(MSVC)
        list(APPEND LAVAAN_COMPILE_OPTIONS /arch:AVX2)
    else()
        list(APPEND LAVAAN_COMPILE_OPTIONS -mavx2 -mfma)
    endif()
    list(APPEND LAVAAN_COMPILE_DEFINITIONS lavaan_AVX2_ENABLED)
endif()

# OpenMP
if(lavaan_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        list(APPEND LAVAAN_COMPILE_DEFINITIONS lavaan_OPENMP_ENABLED)
    else()
        message(WARNING "OpenMP requested but not found")
    endif()
endif()

# ============================================================================
# Dependencies
# ============================================================================
find_package(fmt CONFIG QUIET)
find_package(nlohmann_json CONFIG QUIET)

if(lavaan_BUILD_BENCHMARKS)
    find_package(benchmark CONFIG QUIET)

    if(NOT TARGET benchmark::benchmark)
        include(FetchContent)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(
                google_benchmark
                GIT_REPOSITORY https://github.com/google/benchmark.git
                GIT_TAG v1.9.1
        )
        FetchContent_MakeAvailable(google_benchmark)
    endif()
endif()

# std::linalg is a C++26 header. Rtools GCC 14 does not ship it yet, so use the
# P1673/stdBLAS reference implementation until the active standard library does.
if(NOT TARGET std::linalg)
    include(CheckIncludeFileCXX)
    check_include_file_cxx("linalg" LAVAAN_HAS_NATIVE_LINALG_HEADER)

    if(LAVAAN_HAS_NATIVE_LINALG_HEADER)
        add_library(lavaan_native_linalg INTERFACE)
        add_library(std::linalg ALIAS lavaan_native_linalg)
    endif()
endif()

if(NOT TARGET std::linalg)
    find_package(LinAlg CONFIG QUIET)
endif()

if(NOT TARGET std::linalg)
    include(FetchContent)
    set(LINALG_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(LINALG_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LINALG_ENABLE_BLAS OFF CACHE BOOL "" FORCE)
    set(LINALG_ENABLE_TBB OFF CACHE BOOL "" FORCE)
    set(LINALG_ENABLE_KOKKOS OFF CACHE BOOL "" FORCE)
    # stdBLAS itself only advertises CMake standard selections through C++23.
    # The lavaan targets still build as C++26 below.
    set(LINALG_CXX_STANDARD 23 CACHE STRING "" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_BLAS TRUE CACHE BOOL "" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_TBB TRUE CACHE BOOL "" FORCE)
    set(CMAKE_DISABLE_FIND_PACKAGE_KokkosKernels TRUE CACHE BOOL "" FORCE)

    FetchContent_Declare(
            stdblas
            GIT_REPOSITORY https://github.com/kokkos/stdBLAS.git
            GIT_TAG 70c5127703348209567640c45492be290f6a01ad
    )
    FetchContent_MakeAvailable(stdblas)

    if(TARGET linalg)
        target_include_directories(linalg INTERFACE
                $<BUILD_INTERFACE:${stdblas_BINARY_DIR}/include/experimental>
        )
    endif()
endif()

# ============================================================================
# lavaan Library
# ============================================================================
add_library(lavaan STATIC
        src/library.cpp
        src/library.h
        src/include/utils.h
)
target_include_directories(lavaan PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_options(lavaan PRIVATE ${LAVAAN_COMPILE_OPTIONS})
target_compile_definitions(lavaan PUBLIC ${LAVAAN_COMPILE_DEFINITIONS})
target_compile_features(lavaan PUBLIC cxx_std_26)

if(fmt_FOUND)
    target_link_libraries(lavaan PUBLIC fmt::fmt)
endif()

target_link_libraries(lavaan PUBLIC std::linalg)

if(OpenMP_CXX_FOUND AND lavaan_ENABLE_OPENMP)
    target_link_libraries(lavaan PUBLIC OpenMP::OpenMP_CXX)
endif()

# Alias for consistent naming
add_library(lavaan::lavaan ALIAS lavaan)

# ============================================================================
# Tests
# ============================================================================
if(lavaan_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(lavaan_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "lavaan Configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "  AVX2 enabled:   ${lavaan_ENABLE_AVX2}")
message(STATUS "  OpenMP found:   ${OpenMP_CXX_FOUND}")
message(STATUS "  Build tests:    ${lavaan_BUILD_TESTS}")
message(STATUS "  Build benchmarks: ${lavaan_BUILD_BENCHMARKS}")
message(STATUS "")
