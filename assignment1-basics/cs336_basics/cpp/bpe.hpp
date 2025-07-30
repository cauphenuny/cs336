#include "utils/thread.hpp"

#include <algorithm>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <print>

namespace py = pybind11;

namespace bpe {

inline auto merge_token(
    const std::vector<std::string>& tokens,
    const std::pair<std::string, std::string>& merge_pair,
    std::map<std::pair<std::string, std::string>, int>& count) {
    std::vector<std::string> new_tokens;
    const std::string& first = merge_pair.first;
    const std::string& second = merge_pair.second;
    std::string concat = first + second;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (i + 1 < tokens.size() && tokens[i] == first && tokens[i + 1] == second) {
            new_tokens.push_back(first + second);
            i++;
        } else {
            new_tokens.push_back(tokens[i]);
        }
    }
    count.clear();
    for (size_t i = 0; i + 1 < new_tokens.size(); i++) {
        count[std::make_pair(new_tokens[i], new_tokens[i + 1])]++;
    }
    return new_tokens;
}

inline auto
encode(const py::list& words, const py::list& merges, const py::dict& vocab, int num_threads)
    -> std::vector<int> {
    std::vector<int> token_ids;
    std::vector<std::vector<std::string>> words_vec, merged_words;
    std::map<std::pair<std::string, std::string>, int> merges_rank;
    // int flag = 0;
    for (size_t rank = 0; const auto& item : merges) {
        py::tuple merge = item.cast<py::tuple>();
        std::string first = py::bytes(merge[0]).cast<std::string>(), second = py::bytes(merge[1]).cast<std::string>();
        merges_rank[std::make_pair(first, second)] = rank++;
        // std::println("{}: {}-{}", rank, first, second);
        // if (first == "l" && second == "lo") flag = 1;
    }
    for (auto item : words) {
        py::tuple word = item.cast<py::tuple>();
        std::vector<std::string> word_tokens;
        for (auto token : word) {
            word_tokens.push_back(token.cast<std::string>());
        }
        words_vec.push_back(word_tokens);
    }
    merged_words.resize(words_vec.size());
    // #pragma omp parallel for
    transform(words_vec, [merges_rank](std::vector<std::string> word) {
        std::map<std::pair<std::string, std::string>, int> counts;
        for (size_t i = 0; i + 1 < word.size(); i++) {
            auto pair = std::pair{word[i], word[i + 1]};
            counts[pair]++;
        }
        std::vector<std::pair<std::string, std::string>> valid_pairs;
        while (counts.size()) {
            valid_pairs.clear();
            // std::println("{}", word);
            for (auto&& [pair, count] : counts) {
                // std::println("pair: {}, count: {}, rank: {}", pair, count, merges_rank.contains(pair) ? merges_rank[pair] : -1);
                if (count && merges_rank.contains(pair)) valid_pairs.push_back(pair);
            }
            // std::println("{}, {}", word, valid_pairs);
            if (!valid_pairs.size()) break;
            std::sort(valid_pairs.begin(), valid_pairs.end(), [&merges_rank](const auto& a, const auto& b) -> bool {
                return merges_rank.at(a) < merges_rank.at(b);
            });
            // std::println("{}, merge {}", valid_pairs, valid_pairs[0]);
            // getchar();
            word = merge_token(word, valid_pairs[0], counts);
        }
        // std::println("word: {}", word);
        return word;
    },
    num_threads);
    // 1);
    // std::println("words_vec: {}", words_vec);
    for (const auto& word : words_vec) {
        for (const auto& token : word) {
            py::bytes token_bytes = py::bytes(token);
            if (vocab.contains(token_bytes)) {
                token_ids.push_back(vocab[token_bytes].cast<int>());
            } else {
                throw std::runtime_error("Token not found in vocabulary: " + token);
            }
        }
    }
    return token_ids;
}

}  // namespace bpe
