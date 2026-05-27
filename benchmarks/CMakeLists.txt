add_executable(lavaan_benchmarks
        library_benchmark.cpp
)

target_link_libraries(lavaan_benchmarks PRIVATE
        lavaan::lavaan
        benchmark::benchmark
)

target_compile_options(lavaan_benchmarks PRIVATE ${LAVAAN_COMPILE_OPTIONS})
target_compile_features(lavaan_benchmarks PRIVATE cxx_std_26)
