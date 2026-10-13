cmake_minimum_required(VERSION 3.24)
project(spark_kernels LANGUAGES CXX CUDA)

# Primary target: RTX 5090, compute capability 12.0 (sm_120). The DGX Spark (GB10) is 12.1:
# pass -DCMAKE_CUDA_ARCHITECTURES=121, or "120;121" for one build that runs on both.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit 13.0 REQUIRED)

option(SPARK_BUILD_BENCH "Build benchmark binaries (needs cuBLAS)" ON)
option(SPARK_LINEINFO "Embed line info for Nsight Compute source view" ON)
option(SPARK_BUILD_CPU "Build the CPU-side ops (spark_cpu) and bench_cpu" ON)

# The architecture-specific form of each requested arch ("120a"): the block-scaled fp8
# mma.sync (.kind::mxf8f6f4) that gives the fp8 GEMM its full rate, and the fp4 one
# (.kind::mxf4nvf4) that is the fp4 GEMM's only instruction, are only exposed there
# (docs/design/fp8gemm.md, docs/design/fp4gemm.md). Only the fp8 kernels (the GEMM and the
# fp8 attention), the fp4 kernels and bench_peak are built for it; an "a" cubin runs on
# exactly that compute capability, which the plain arch already implies here.
set(SPARK_CUDA_ARCHITECTURES_A "")
foreach(arch ${CMAKE_CUDA_ARCHITECTURES})
  if(arch MATCHES "a$")
    list(APPEND SPARK_CUDA_ARCHITECTURES_A "${arch}")
  else()
    list(APPEND SPARK_CUDA_ARCHITECTURES_A "${arch}a")
  endif()
endforeach()

# ---- kernel library ----------------------------------------------------------------------
file(GLOB SPARK_KERNEL_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/*.cu)
file(GLOB SPARK_FP8_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/kernels/fp8gemm*.cu
     ${CMAKE_SOURCE_DIR}/src/kernels/attention_fp8*.cu
     ${CMAKE_SOURCE_DIR}/src/kernels/fp8quant.cu
     ${CMAKE_SOURCE_DIR}/src/kernels/fp4*.cu
     ${CMAKE_SOURCE_DIR}/src/kernels/grouped_gemm*.cu)
list(REMOVE_ITEM SPARK_KERNEL_SOURCES ${SPARK_FP8_SOURCES})
add_library(spark_kernels_fp8 STATIC ${SPARK_FP8_SOURCES})
add_library(spark_kernels STATIC ${SPARK_KERNEL_SOURCES})
target_link_libraries(spark_kernels PUBLIC spark_kernels_fp8)
set_target_properties(spark_kernels_fp8 PROPERTIES CUDA_ARCHITECTURES "${SPARK_CUDA_ARCHITECTURES_A}")
foreach(lib spark_kernels spark_kernels_fp8)
  target_include_directories(${lib} PUBLIC ${CMAKE_SOURCE_DIR}/include)
  target_compile_options(${lib} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -Xcompiler=-Wall>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${SPARK_LINEINFO}>>:-lineinfo>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>)
  set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF POSITION_INDEPENDENT_CODE ON)
endforeach()

# ---- benchmark binaries ------------------------------------------------------------------
if(SPARK_BUILD_BENCH)
  file(GLOB SPARK_BENCH_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/bench/bench_*.cu)
  foreach(src ${SPARK_BENCH_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src/bench)
    # cublasLt is only for bench_fp8gemm's reference (cuBLAS has no fp8 GemmEx path).
    target_link_libraries(${name} PRIVATE spark_kernels CUDA::cublas CUDA::cublasLt CUDA::cudart)
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${SPARK_LINEINFO}>>:-lineinfo>)
  endforeach()
  # bench_peak measures the block-scaled fp8 instruction, so it needs the "a" arch too.
  set_target_properties(bench_peak PROPERTIES CUDA_ARCHITECTURES "${SPARK_CUDA_ARCHITECTURES_A}")
endif()

# ---- CPU-side ops (the GB10's Arm cores) ----------------------------------------------------
# include/spark/cpu.h and src/cpu/*.cpp: a 1-bit GEMV, routing, sampling and a bandwidth
# probe in plain C++17, no CUDA. On aarch64 (the DGX Spark) it is built for Arm v9 with SVE2
# so the NEON paths and the dot-product extension are in; elsewhere only the scalar rungs
# compile (the NEON code sits behind __ARM_NEON). bench_cpu links against spark_cpu alone.
if(SPARK_BUILD_CPU)
  file(GLOB SPARK_CPU_SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/cpu/*.cpp)
  list(REMOVE_ITEM SPARK_CPU_SOURCES ${CMAKE_SOURCE_DIR}/src/cpu/bench_cpu.cpp)
  add_library(spark_cpu STATIC ${SPARK_CPU_SOURCES})
  target_include_directories(spark_cpu PUBLIC ${CMAKE_SOURCE_DIR}/include)
  set_target_properties(spark_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
  find_package(Threads REQUIRED)
  target_link_libraries(spark_cpu PUBLIC Threads::Threads)
  set(SPARK_CPU_FLAGS -O3 -Wall)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$" AND NOT APPLE)
    list(APPEND SPARK_CPU_FLAGS -march=armv9-a+sve2)
  endif()
  target_compile_options(spark_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${SPARK_CPU_FLAGS}>)
  add_executable(bench_cpu ${CMAKE_SOURCE_DIR}/src/cpu/bench_cpu.cpp)
  target_link_libraries(bench_cpu PRIVATE spark_cpu)
  target_compile_options(bench_cpu PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${SPARK_CPU_FLAGS}>)
endif()
