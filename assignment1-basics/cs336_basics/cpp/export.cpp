#include "bpe.hpp"

#include <iostream>
#include <pybind11/pybind11.h>

using namespace pybind11::literals;  // for _a suffix

void hello() { std::cout << "Hello from the export function!" << std::endl; }

PYBIND11_MODULE(cpp_extensions, m) {
    m.def("hello", &hello, "A function that prints a hello message");
    m.def(
        "encode_bpe", &bpe::encode, "Encode a list of words using merges and vocabulary using BPE",
        "input"_a, "merges"_a, "vocab"_a, "num_threads"_a, "verbose"_a = false);
    m.def("train_bpe", &bpe::train, "Train BPE", "vocab"_a, "word_counts"_a, "pair_counts"_a, "vocab_size"_a);
}
