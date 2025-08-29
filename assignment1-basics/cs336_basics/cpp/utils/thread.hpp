#pragma once
#include "../include/indicators.hpp"

#include <cstddef>
// #include <format>
// #include <iostream>
#include <atomic>
#include <iterator>
#include <thread>
#include <vector>

template <typename Container, typename Func>
void transform(
    Container& container, Func func, size_t num_threads = 1, bool verbose = false,
    std::string_view name = "transform") {
    if (num_threads <= 1 || std::size(container) <= 1) {
        for (auto& item : container) {
            item = func(item);
        }
        return;
    }

    auto begin = std::begin(container);
    auto end = std::end(container);
    // std::cout << std::format("begin - end: {}", end - begin) << std::endl;
    size_t total = std::distance(begin, end);
    size_t chunk_size = (total + num_threads - 1) / num_threads;
    std::vector<std::thread> threads;
    std::atomic<size_t> progress{0};
    for (size_t t = 0; t < num_threads; ++t) {
        auto chunk_begin = std::next(begin, t * chunk_size);
        auto chunk_end =
            (t == num_threads - 1) ? end : std::next(begin, std::min((t + 1) * chunk_size, total));
        // std::cout << std::format("chunk_size: {}", chunk_end - chunk_begin) << std::endl;
        if (std::distance(chunk_begin, chunk_end) <= 0) continue;
        threads.emplace_back([chunk_begin, chunk_end, &func, &progress, verbose]() {
            for (auto it = chunk_begin; it != chunk_end; ++it) {
                *it = func(*it);
                if (verbose) progress++;
            }
        });
    }
    if (verbose) {
        using namespace indicators;
        BlockProgressBar bar{
            option::BarWidth{80},
            option::Start{"["},
            option::End{"]"},
            option::ForegroundColor{Color::unspecified},
            option::ShowPercentage{true},
            option::ShowElapsedTime{true},
            option::ShowRemainingTime{true},
            option::PrefixText{name},
            option::Stream{std::cerr},
        };
        do {
            bar.set_progress(progress * 100.0 / total);
        } while (progress < total);
        bar.set_progress(100);
        std::cerr << std::endl;
    }
    for (auto& th : threads) {
        th.join();
    }
}
