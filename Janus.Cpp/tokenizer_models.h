#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <utility> // for pair
#include <regex>
#include <queue>

#include <vector>
#include <string>
#include <unordered_map>

class TrieNode {
public:
	std::unordered_map<uint8_t, TrieNode*> children;
	bool is_end;
	int value;

	TrieNode() : is_end(false), value(0) {}
};

class Trie {
private:
	TrieNode* root;

public:
	Trie() : root(new TrieNode()) {}

	void insert(const std::vector<uint8_t>& key, const int value) {
		TrieNode* node = root;
		for (char c : key) {
			if (node->children.find(c) == node->children.end()) {
				node->children[c] = new TrieNode();
			}
			node = node->children[c];
		}
		node->is_end = true;
		node->value = value;
	}

	std::pair<int, int> searchLongest(const std::vector<uint8_t>& s, int start) const {
		TrieNode* node = root;
		int max_len = 0;
		int found_value = 0;
		for (int i = start; i < s.size(); ++i) {
			char c = s[i];
			auto it = node->children.find(c);
			if (it == node->children.end()) {
				break;
			}
			node = it->second;
			if (node->is_end) {
				int current_len = i - start + 1;
				if (current_len > max_len) {
					max_len = current_len;
					found_value = node->value;
				}
			}
		}
		return { max_len, found_value };
	}
};

class Replacer {
private:
	Trie trie;

public:
	void load_vocab(const std::vector<std::vector<uint8_t>>& replacements) {
		for (const auto& [i, str] : replacements | std::views::enumerate) {
			trie.insert(str, static_cast<int>(i));
		}
	}

	std::vector<int> replace(const std::vector<uint8_t>& s) const {
		std::vector<int> result;
		int i = 0;
		while (i < s.size()) {
			auto [len, val] = trie.searchLongest(s, i);
			if (len > 0) {
				result.push_back(val);
				i += len;
			}
			else {
				throw std::runtime_error("Unknown token");
			}
		}
		return result;
	}
};

struct BPEModel {
	std::vector<std::vector<uint8_t>> vocab; // token到id的映射
	Replacer vocab_replacer;
	std::vector<std::vector<uint8_t>> special_tokens; // 特殊token到id的映射
	std::vector<std::vector<uint8_t>> byte_map; // 字节到token char的映射
	constexpr static int special_start = 100000;
};