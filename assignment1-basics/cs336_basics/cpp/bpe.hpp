#include "include/indicators.hpp"
#include "utils/thread.hpp"

#include <algorithm>
#include <map>
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace bpe {

struct pair_hash {
    template <class T1, class T2> std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        // 组合两个哈希值
        return h1 ^ (h2 << 1);
    }
};

inline auto merge_token(
    const std::vector<std::string>& tokens, const std::pair<std::string, std::string>& merge_pair) {
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
    return new_tokens;
}

inline auto
encode(const py::list& words, const py::list& merges, const py::dict& vocab, int num_threads)
    -> std::vector<int> {
    std::vector<int> token_ids;
    std::vector<std::vector<std::string>> words_vec, merged_words;
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> merges_rank;
    // int flag = 0;
    for (size_t rank = 0; const auto& item : merges) {
        py::tuple merge = item.cast<py::tuple>();
        std::string first = py::bytes(merge[0]).cast<std::string>(),
                    second = py::bytes(merge[1]).cast<std::string>();
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
    transform(
        words_vec,
        [merges_rank](std::vector<std::string> word) {
            std::vector<std::pair<std::pair<std::string, std::string>, int>> valid_pairs;
            do {
                valid_pairs.clear();
                // std::println("{}", word);
                for (size_t i = 0; i + 1 < word.size(); i++) {
                    auto pair = std::make_pair(word[i], word[i + 1]);
                    if (merges_rank.contains(pair))
                        valid_pairs.push_back(std::make_pair(pair, merges_rank.at(pair)));
                }
                // std::println("{}, {}", word, valid_pairs);
                if (!valid_pairs.size()) break;
                std::sort(
                    valid_pairs.begin(), valid_pairs.end(),
                    [&merges_rank](const auto& a, const auto& b) -> bool {
                        return a.second < b.second;
                    });
                // std::println("{}, merge {}", valid_pairs, valid_pairs[0]);
                // getchar();
                word = merge_token(word, valid_pairs[0].first);
            } while (valid_pairs.size());
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

using Bytes = std::string;  // py::bytes <-> std::string
using Word = std::vector<Bytes>;
using Pair = std::pair<Bytes, Bytes>;

inline auto
train(py::dict vocab_py, py::dict word_counts_py, py::dict pair_counts_py, int vocab_size)
    -> std::tuple<py::dict, py::list> {
    // 1. Python dict -> C++ map
    std::map<Word, int> word_counts;
    for (auto item : word_counts_py) {
        py::tuple word_tuple = item.first.cast<py::tuple>();
        Word word;
        for (auto obj : word_tuple) {
            word.push_back(py::cast<std::string>(obj));
        }
        int count = item.second.cast<int>();
        word_counts[word] = count;
    }

    std::unordered_map<Pair, int, pair_hash> pair_counts;
    for (auto item : pair_counts_py) {
        py::tuple pair_tuple = item.first.cast<py::tuple>();
        Pair pair = {
            py::bytes(pair_tuple[0]).cast<std::string>(),
            py::bytes(pair_tuple[1]).cast<std::string>()};
        int count = item.second.cast<int>();
        pair_counts[pair] = count;
    }

    // 2. vocab, merges
    std::map<int, Bytes> vocab;  // index -> bytes
    for (auto item : vocab_py) {
        auto index = item.first.cast<int>();
        auto bytes = item.second.cast<Bytes>();
        vocab[index] = bytes;
    }
    py::list merges;

    // 3. 主循环
    using namespace indicators;
    BlockProgressBar bar{
        // option::BarWidth{80},
        option::Start{"["},
        option::End{"]"},
        option::ForegroundColor{Color::white},
        option::ShowPercentage{true},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::PrefixText{"Training BPE"},
        // option::FontStyles{std::vector<FontStyle>{FontStyle::bold}},
        option::Stream{std::cerr},
    };
    int cur_progress = -1;

    while (vocab.size() < vocab_size) {

        if (pair_counts.empty()) break;

        while ((int)vocab.size() * 100 / vocab_size > cur_progress) {
            bar.tick();
            cur_progress++;
        }

        // 找出现次数最多的 pair
        auto best_pair = std::max_element(
            pair_counts.begin(), pair_counts.end(), [](const auto& a, const auto& b) {
                if (a.second != b.second)
                    return a.second < b.second;
                else if (a.first.first != b.first.first)
                    return a.first.first < b.first.first;
                else
                    return a.first.second < b.first.second;  // 按字典序比较
            });
        if (best_pair == pair_counts.end()) break;

        Pair merge_pair = best_pair->first;
        int _best_count = best_pair->second;

        // 记录 merge
        merges.append(py::make_tuple(py::bytes(merge_pair.first), py::bytes(merge_pair.second)));
        Bytes new_vocab = merge_pair.first + merge_pair.second;
        vocab[vocab.size()] = new_vocab;

        while (true) {
            // 需要更新 word_counts 和 pair_counts
            std::vector<std::pair<Word, Word>> update_word;
            std::vector<std::tuple<Pair, std::optional<Pair>, int>> update_pair;

            for (auto it = word_counts.begin(); it != word_counts.end();) {
                const Word& word = it->first;
                int count = it->second;
                for (size_t index = 0; index + 1 < word.size(); ++index) {
                    if (std::make_pair(word[index], word[index + 1]) == merge_pair) {
                        // 构造新 word
                        Word new_word;
                        new_word.insert(new_word.end(), word.begin(), word.begin() + index);
                        new_word.push_back(new_vocab);
                        new_word.insert(new_word.end(), word.begin() + index + 2, word.end());
                        update_word.emplace_back(word, new_word);
                        update_pair.emplace_back(merge_pair, std::nullopt, count);
                        if (index > 0) {
                            update_pair.emplace_back(
                                Pair{word[index - 1], word[index]},
                                Pair{word[index - 1], new_vocab}, count);
                        }
                        if (index + 2 < word.size()) {
                            update_pair.emplace_back(
                                Pair{word[index + 1], word[index + 2]},
                                Pair{new_vocab, word[index + 2]}, count);
                        }
                        break;
                    }
                }
                ++it;
            }

            if (update_word.empty()) break;

            // 更新 word_counts
            for (const auto& [word, new_word] : update_word) {
                word_counts[new_word] += word_counts[word];
                word_counts.erase(word);
            }

            // 更新 pair_counts
            for (const auto& [pair, new_pair_opt, count] : update_pair) {
                pair_counts[pair] -= count;
                if (pair_counts[pair] <= 0) pair_counts.erase(pair);
                if (new_pair_opt) {
                    pair_counts[*new_pair_opt] += count;
                }
            }
        }
    }

    // 输出 vocab（index顺序）
    py::dict vocab_dict;
    for (const auto& [idx, val] : vocab) {
        vocab_dict[py::int_(idx)] = py::bytes(val);
    }
    return std::make_tuple(vocab_dict, merges);
}

}  // namespace bpe
