#pragma once

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

#include <vector>
#include <memory>
#include <ranges>
#include <filesystem>
#include <unordered_map>

#include "tensor_utils.h"
#include "quant.h"

constexpr std::array<unsigned, 5> ch_mult{ 1, 1, 2, 2, 4 };

class DataEater
{
private:
	std::unordered_map<
		std::string, std::vector<std::filesystem::path>
	> weight_data_path;
	std::optional<std::filesystem::path> target = std::nullopt;
	const unsigned depth = 0;
	const std::string cur_name;

private:
	DataEater(const std::vector<std::filesystem::path>& paths,
		std::string cur_name, unsigned depth = 1)
		: depth(depth), cur_name(cur_name)
	{
		if (paths.empty())
			throw std::runtime_error("No tensor data found");
		if (paths.size() == 1) {
			target = paths.front();
		}
		for (const auto& path : paths)
		{
			auto filename = path.filename().string();
			weight_data_path[get_name_part_depth(filename)].push_back(path);
		}
	}

	inline DataEater Goto(const auto& key0, const auto&... keys)
	{
		auto eater = operator[](key0);
		if constexpr (sizeof...(keys) > 0)
			return eater.Goto(keys...);
		else
			return eater;
	}

	inline std::string get_name_part_depth(const std::string& name)
	{
		auto iter = name.begin();
		unsigned cur_depth = 0;
		while (iter != name.end() && cur_depth < depth)
		{
			if (*iter == '.')
				cur_depth++;
			iter++;
		}
		auto next_pos = name.find('.', iter - name.begin());
		if (next_pos == std::string::npos)
			return name.substr(iter - name.begin());
		return name.substr(iter - name.begin(),
			next_pos - (iter - name.begin()));
	}
public:
	DataEater(const std::filesystem::path& weight_dir, unsigned depth = 0)
		: depth(depth), cur_name("")
	{
		for (const auto& path :
			std::filesystem::directory_iterator(weight_dir)
			| std::views::filter([](const auto& entry) {
				return entry.is_regular_file();
				})
			| std::views::transform([](const auto& entry) {
				return entry.path();
				})
			)
		{
			auto filename = path.filename().string();
			weight_data_path[get_name_part_depth(filename)].push_back(path);
		}
	}

	inline DataEater operator[](const size_t key) {
		return operator[](std::to_string(key));
	}

	inline DataEater operator[](const std::string& key) {
		if (auto it = weight_data_path.find(key);
			it != weight_data_path.end())
			return DataEater(it->second, cur_name + "." + key, depth + 1);
		throw std::runtime_error("No tensor data found");
	}

	template <typename Key0, typename... Keys>
	inline DataEater operator()(Key0 key0, Keys... keys) {
		return Goto(key0, keys...);
	}

	inline std::vector<uint8_t> ReadTensorData(ggml_type type)
	{
		if (!target)
			throw std::runtime_error("No target tensor data found");
		if (type == GGML_TYPE_F16)
			return F16DataFromFile(target.value());
		else if (type == GGML_TYPE_F32)
			return F32DataFromFile(target.value());
		throw std::runtime_error("Unsupported data type");
	}

	inline void LoadToBackend(ggml_tensor* out)
	{
		auto data = ReadTensorData(out->type);
		if (data.size() != ggml_nbytes(out))
			throw std::runtime_error("Data size mismatch");
		ggml_backend_tensor_set(out, data.data(), 0, data.size());
	}

	inline const std::string& GetName() const {
		return cur_name;
	}
};

class Convolution
{
private:
	const unsigned cin;
	const unsigned cout;
	const unsigned kernel_size = 3;
	ggml_context* block_ctx;

public:
	Convolution(
		unsigned ch_in, unsigned ch_out,
		unsigned kernel_size,
		DataEater weight_data,
		ggml_backend* backend
	);

	inline ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = ggml_conv_2d_s1_ph(ctx, conv, input);
		return ggml_add_inplace(ctx, x, conv_bias);
	}

private:
	ggml_tensor* conv;
	ggml_tensor* conv_bias;
};

class GroupNorm
{
private:
	const unsigned cin;
	const unsigned group_norm_n_groups = 32;
	ggml_context* block_ctx;

public:
	GroupNorm(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	);

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input, bool inplace);

private:
	ggml_tensor* weight;
	ggml_tensor* bias;
};

class ResNetBlock
{
private:
	const unsigned cin;
	const unsigned cout;
	const unsigned group_norm_n_groups = 32;
	const unsigned kernel_size = 3;
	const std::string name;

public:
	ResNetBlock(
		unsigned ch_in, unsigned ch_out,
		DataEater weight_data,
		ggml_backend* backend
	);

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input);

private:
	GroupNorm norm1, norm2;
	Convolution conv_in, conv_out;
	std::optional<Convolution> nin_shortcut;
};

class GroupedResBlock
{
private:
	const unsigned cin;
	const unsigned cout;
	const unsigned n_blocks;
public:
	GroupedResBlock(
		unsigned ch_in, unsigned ch_out,
		DataEater weight_data,
		ggml_backend* backend,
		unsigned n_blocks
	);

	inline ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = input;
		for (auto& resblock : resblocks)
			x = resblock(ctx, x);
		return x;
	}

	inline const auto& GetBlocks() const {
		return resblocks;
	}
	inline const auto& operator[](size_t idx) const {
		return resblocks[idx];
	}
private:
	std::vector<ResNetBlock> resblocks;
};

class AttnBlock
{
private:
	const unsigned cin;
	const unsigned group_norm_n_groups = 32;
	std::string name;

public:
	AttnBlock(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	);

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input);

private:
	GroupNorm norm;
	Convolution q_proj, k_proj, v_proj, o_proj;
};

class Downsample
{
private:
	const unsigned cin;

public:
	Downsample(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	)
		: cin(ch_in),
		conv{ ch_in, ch_in, 3, weight_data["conv"], backend }
	{
	}

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input) {
		auto width = input->ne[0], height = input->ne[1];
		auto padded = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
			width + 1, height + 1, cin, input->ne[3]);
		padded = ggml_scale_inplace(ctx, padded, 0.f);
		padded = ggml_set_2d_inplace(ctx, padded, input,
			4 * (width + 1), 4 * (width + 2));
		return conv(ctx, padded);
	}

private:
	Convolution conv;
};

class Upsample
{
private:
	const unsigned cin;

public:
	Upsample(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	)
		: cin(ch_in),
		conv{ ch_in, ch_in, 3, weight_data["conv"], backend }
	{
	}

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input) {
		auto width = input->ne[0], height = input->ne[1];
		auto x = ggml_upscale(ctx, input, 2);
		return conv(ctx, x);
	}
private:
	Convolution conv;
};

class VQ_Encoder
{
private:
	const unsigned in_channels = 3;
	const unsigned resblock_ch = 128;
	const unsigned z_channels = 256;

public:
	VQ_Encoder(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	);

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input);

private:
	Convolution conv_in, conv_out;
	GroupNorm norm_out;

	std::vector<GroupedResBlock> res_blocks;
	std::vector<Downsample> down_samples;
	std::vector<AttnBlock> attn_blocks;

	ResNetBlock midres1, midres2;
	AttnBlock midattn;
};

class VQ_Quantizer
{
private:
	ggml_context* quant_ctx;
public:
	VQ_Quantizer(
		DataEater weight_data,
		ggml_backend* backend
	);

	~VQ_Quantizer() {
		ggml_free(quant_ctx);
	}

	ggml_tensor* quantize(ggml_context* ctx, ggml_tensor* z);

	inline ggml_tensor* get_rows(ggml_context* ctx, ggml_tensor* idx) {
		return ggml_get_rows(ctx, embedding, idx);
	}
private:
	ggml_tensor* embedding;
};

// Data Tree :
// decoder
//  mid
//   res1
//   attn
//   res2
// conv_blocks
//   [i]
//    res x 3
//    attn x 3 (optional)
//    upsample (optional)
//
class VQ_Decoder
{
private:
	const unsigned resblock_ch = 128;
	const unsigned z_channels = 256;

public:
	VQ_Decoder(
		unsigned ch_out,
		DataEater block_data,
		ggml_backend* backend
	);

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* z);

private:
	Convolution conv_in, conv_out;
	GroupNorm norm_out;

	ResNetBlock midres1, midres2;
	AttnBlock midattn;

	std::vector<AttnBlock> attn_blocks;
	std::vector<GroupedResBlock> res_blocks;
	std::vector<Upsample> up_samples;
};

class GenDecoder
{
private:
	const unsigned quant_channels = 8;
	ggml_backend* backend;
	const std::filesystem::path folder = R"(D:\Python\Janus\model-file\vq)";
	DataEater weight_data;

public:
	GenDecoder(
		ggml_backend* backend
	);

	std::vector<uint8_t> decode_img_tokens(
		std::vector<int> tokens, size_t parallel_size, size_t patch_nums);

private:
	Convolution quant_conv;
	Convolution post_quant_conv;

	VQ_Quantizer quant;
	VQ_Decoder dec;
};
