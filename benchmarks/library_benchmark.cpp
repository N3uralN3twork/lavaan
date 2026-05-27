#include <benchmark/benchmark.h>

#include "library.h"

#include <functional>
#include <iostream>
#include <streambuf>
#include <type_traits>

namespace {

class NullStreamBuffer final : public std::streambuf {
protected:
    std::streamsize xsputn(const char*, std::streamsize count) override
    {
        return count;
    }

    int overflow(int character) override
    {
        return traits_type::not_eof(character);
    }
};

class ScopedCoutSilencer {
public:
    ScopedCoutSilencer()
        : previous_buffer_(std::cout.rdbuf(&buffer_))
    {
    }

    ~ScopedCoutSilencer()
    {
        std::cout.rdbuf(previous_buffer_);
    }

    ScopedCoutSilencer(const ScopedCoutSilencer&) = delete;
    ScopedCoutSilencer& operator=(const ScopedCoutSilencer&) = delete;

private:
    NullStreamBuffer buffer_;
    std::streambuf* previous_buffer_;
};

template <typename Function>
void benchmark_function(benchmark::State& state, Function&& function)
{
    for (auto _ : state) {
        if constexpr (std::is_void_v<std::invoke_result_t<Function&>>) {
            std::invoke(function);
            benchmark::ClobberMemory();
        } else {
            benchmark::DoNotOptimize(std::invoke(function));
        }
    }
}

void BM_Hello(benchmark::State& state)
{
    ScopedCoutSilencer silence;
    benchmark_function(state, [] {
        hello();
    });
}

} // namespace

BENCHMARK(BM_Hello);

BENCHMARK_MAIN();
