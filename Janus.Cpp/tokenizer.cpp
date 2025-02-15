#include <sstream>
#include <algorithm>
#include <ranges>

#include "tokenizer.h"

BPEModel load_bpe_model(std::filesystem::path model_folder)
{
	BPEModel model;
	auto vocab = model_folder / "vocab.txt";
	auto byte_map = model_folder / "byte_map.txt";
	auto special_tokens = model_folder / "special_tokens.txt";

	std::ifstream fp;
	std::string line;

	// 解析词汇表
	fp.open(vocab);
	if (fp.good())
	{
		int index = 0;
		while (std::getline(fp, line)) {
			std::vector<uint8_t> vec(line.begin(), line.end());
			model.vocab.push_back(vec);
		}
	}
	else
		throw std::runtime_error("无法打开词汇表文件");
	model.vocab_replacer.load_vocab(model.vocab);

	fp.close();
	fp.open(byte_map);
	if (fp.good())
	{
		while (std::getline(fp, line)) {
			std::vector<uint8_t> vec(line.begin(), line.end());
			model.byte_map.push_back(vec);
		}
	}
	else
		throw std::runtime_error("无法打开字节映射文件");

	fp.close();
	fp.open(special_tokens);
	if (fp.good())
	{
		while (std::getline(fp, line)) {
			std::vector<uint8_t> vec(line.begin(), line.end());
			model.special_tokens.push_back(vec);
		}
	}
	else
		throw std::runtime_error("无法打开特殊token文件");

	return model;
}

std::vector<int> tokenizer_encode(const BPEModel& model, std::string raw_text)
{
	std::vector<uint8_t> text(raw_text.begin(), raw_text.end());
	// 将文本转换为字节序列（UTF-8编码）
	std::vector<std::pair<size_t, int>> special_positions;
	int special_id = BPEModel::special_start;
	for (auto spe : model.special_tokens)
	{
		auto pos = std::search(text.begin(), text.end(), spe.begin(), spe.end());
		while (pos != text.end())
		{
			special_positions.push_back({ pos - text.begin(), special_id});
			pos = std::search(pos + 1, text.end(), spe.begin(), spe.end());
		}
		special_id++;
	}
	std::sort(special_positions.begin(), special_positions.end());
	std::vector<std::vector<uint8_t>> splitted_text;
	size_t last_pos = 0;
	for (const auto [pos, tid] : special_positions)
	{
		if (pos > last_pos)
			splitted_text.emplace_back(text.begin() + last_pos, text.begin() + pos);
		last_pos = pos + model.special_tokens[tid - BPEModel::special_start].size();
	}
	if (last_pos < text.size())
		splitted_text.emplace_back(text.begin() + last_pos, text.end());

	std::vector<int> result;
	for (const auto [i, text] : splitted_text | std::views::enumerate)
	{
		if (i > 0)
			result.push_back(special_positions[i - 1].second);
		std::vector<uint8_t> mapped_text;
		for (char c : text)
		{
			int byte = (unsigned char)c;
			mapped_text.append_range(model.byte_map[byte]);
		}
		result.append_range(model.vocab_replacer.replace(mapped_text));
	}

	return result;
}

std::string tokenizer_decode(const BPEModel& model, const std::vector<int>& tokens)
{
	std::vector<std::vector<uint8_t>> splitted_text;
	std::vector<uint8_t> current_text;
	std::vector<int> specials;
	for (int token : tokens)
	{
		if (token >= BPEModel::special_start)
		{
			splitted_text.push_back(current_text);
			current_text.clear();
			specials.push_back(token);
		}
		else
			current_text.append_range(model.vocab[token]);
	}
	splitted_text.push_back(current_text);

	for (auto& text : splitted_text) {
		for (auto [i, code] : model.byte_map | std::views::enumerate)
		{
			if (i == code[0] && code.size() == 1)
				continue;
			auto pos = std::search(text.begin(), text.end(), code.begin(), code.end());
			while (pos != text.end())
			{
				auto szpos = std::distance(text.begin(), pos);
				if (code.size() > 1)
					text.erase(pos, pos + code.size() - 1);
				text[szpos] = (uint8_t)i;
				pos = std::search(text.begin(), text.end(), code.begin(), code.end());
			}
		}
	}

	std::vector<uint8_t> result;
	for (int i = 0; i < splitted_text.size(); i++)
	{
		result.append_range(splitted_text[i]);
		if (i < specials.size())
			result.append_range(model.vocab[specials[i]]);
	}

	std::string result_text;
	for (uint8_t c : result)
		result_text.push_back((char)c);

	return result_text;
}