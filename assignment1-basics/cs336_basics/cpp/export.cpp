#include "bpe.hpp"

#include <iostream>
#include <pybind11/pybind11.h>

void hello() { std::cout << "Hello from the export function!" << std::endl; }

PYBIND11_MODULE(cpp_extensions, m) {
    m.def("hello", &hello, "A function that prints a hello message");
    m.def(
        "encode_bpe", &bpe::encode, "Encode a list of words using merges and vocabulary using BPE");
    m.def(
        "train_bpe", &bpe::train, "Train BPE"
    );
}
