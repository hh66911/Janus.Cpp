#pragma once

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

#include <vector>
#include <memory>

#include "tensor_utils.h"

const std::array<unsigned, 5> ch_mult{ 1, 1, 2, 2, 4 };

class DataEater
{
private:
	std::unordered_map<
		std::string, std::vector<std::filesystem::path>
	> weight_data_path;
	std::optional<std::filesystem::path> target = std::nullopt;
	const unsigned depth = 0;

private:
	DataEater(const std::vector<std::filesystem::path>& paths, unsigned depth = 1)
		: depth(depth)
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
		: depth(depth)
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
			return DataEater(it->second, depth + 1);
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
	}

	inline void LoadToBackend(ggml_tensor* out)
	{
		auto data = ReadTensorData(out->type);
		if (data.size() != ggml_nbytes(out))
			throw std::runtime_error("Data size mismatch");
		ggml_backend_tensor_set(out, data.data(), 0, data.size());
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
		DataEater weight_data,
		ggml_backend* backend
	)
		: cin(ch_in), cout(ch_out)
	{
		ggml_init_params block_params = {
			.mem_size = ggml_tensor_overhead() * 2,
			.no_alloc = true
		};
		block_ctx = ggml_init(block_params);
		conv = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			kernel_size, kernel_size, ch_in, ch_out);
		conv_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F32, 1, 1, ch_out);
		ggml_backend_alloc_ctx_tensors(block_ctx, backend);
		weight_data["weight"].LoadToBackend(conv);
		weight_data["bias"].LoadToBackend(conv_bias);
	}

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input)
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
	)
		: cin(ch_in)
	{
		ggml_init_params block_params = {
			.mem_size = ggml_tensor_overhead() * 2,
			.no_alloc = true
		};
		block_ctx = ggml_init(block_params);
		norm = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		norm_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F32, 1, 1, ch_in);
		ggml_backend_alloc_ctx_tensors(block_ctx, backend);
		weight_data["norm"].LoadToBackend(norm);
		weight_data["norm_bias"].LoadToBackend(norm_bias);
	}
	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = ggml_group_norm(ctx, input, group_norm_n_groups, 1e-6f);
		x = ggml_add_inplace(ctx, ggml_mul_inplace(ctx, x, norm), norm_bias);
		return ggml_sigmoid_inplace(ctx, x);
	}

private:
	ggml_tensor* norm;
	ggml_tensor* norm_bias;
};

class ResNetBlock
{
private:
	const unsigned cin;
	const unsigned cout;
	const unsigned group_norm_n_groups = 32;
	const unsigned kernel_size = 3;

	ggml_context* block_ctx;

public:
	ResNetBlock(
		unsigned ch_in, unsigned ch_out,
		DataEater weight_data,
		ggml_backend* backend
	)
		: cin(ch_in), cout(ch_out),
		norm1{ ch_in, weight_data["norm1"], backend },
		conv_in{ ch_in, ch_out, weight_data["conv_in"], backend },
		norm2{ ch_out, weight_data["norm2"], backend },
		conv_out{ ch_out, ch_out, weight_data["conv_out"], backend }
	{
		if (ch_in != ch_out)
		{

		}
	};

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = norm1(ctx, input);
		x = ggml_sigmoid_inplace(ctx, x);
		x = conv_in(ctx, x);
		x = norm2(ctx, x);
		x = ggml_sigmoid_inplace(ctx, x);
		x = conv_out(ctx, x);
		if (nin_shortcut)
			input = (*nin_shortcut)(ctx, input);
		return ggml_add_inplace(ctx, x, input);
	}

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
	) : cin(ch_in), cout(ch_out),
		n_blocks(n_blocks)
	{
		resblocks.reserve(n_blocks);
		resblocks.emplace_back(ch_in, ch_out,
			weight_data[0], backend);
		for (size_t i = 1; i < n_blocks; i++) {
			resblocks.emplace_back(ch_out, ch_out,
				weight_data[i], backend);
		}
	}

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input) {
		auto x = input;
		for (auto& resblock : resblocks)
			x = resblock.operator()(ctx, x);
		return x;
	}
private:
	std::vector<ResNetBlock> resblocks;
};

class AttnBlock
{
private:
	const unsigned cin;
	const unsigned group_norm_n_groups = 32;

	ggml_context* block_ctx;

public:
	AttnBlock(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	) : cin(ch_in)
	{
		const size_t tensor_num = 10;
		ggml_init_params block_params = {
			.mem_size = ggml_tensor_overhead() * tensor_num,
			.no_alloc = true
		};
		block_ctx = ggml_init(block_params);
		norm = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		norm_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		q_proj = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			1, 1, ch_in, ch_in);
		k_proj = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			1, 1, ch_in, ch_in);
		v_proj = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			1, 1, ch_in, ch_in);
		o_proj = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			1, 1, ch_in, ch_in);
		q_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		k_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		v_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		o_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		ggml_backend_alloc_ctx_tensors(block_ctx, backend);

		weight_data["norm"].LoadToBackend(norm);
		weight_data["norm_bias"].LoadToBackend(norm_bias);
		weight_data["q_proj"].LoadToBackend(q_proj);
		weight_data["k_proj"].LoadToBackend(k_proj);
		weight_data["v_proj"].LoadToBackend(v_proj);
		weight_data["o_proj"].LoadToBackend(o_proj);
		weight_data["q_bias"].LoadToBackend(q_bias);
		weight_data["k_bias"].LoadToBackend(k_bias);
		weight_data["v_bias"].LoadToBackend(v_bias);
		weight_data["o_bias"].LoadToBackend(o_bias);
	}

	~AttnBlock() {
		ggml_free(block_ctx);
	}

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = ggml_group_norm(ctx, input, group_norm_n_groups, 1e-6f);
		x = ggml_add_inplace(ctx, ggml_mul_inplace(ctx, x, norm), norm_bias);

		auto q = ggml_conv_2d_s1_ph(ctx, q_proj, x);
		auto k = ggml_conv_2d_s1_ph(ctx, k_proj, x);
		auto v = ggml_conv_2d_s1_ph(ctx, v_proj, x);
		q = ggml_add_inplace(ctx, q, q_bias);
		k = ggml_add_inplace(ctx, k, k_bias);
		v = ggml_add_inplace(ctx, v, v_bias);

		size_t width = x->ne[0], height = x->ne[1];
		size_t batch = x->ne[3];
		q = view_tensor(ctx, q, 1ull, width * height, cin, batch);
		k = view_tensor(ctx, k, 1ull, width * height, cin, batch);
		auto QK = ggml_mul_mat(ctx, k, q); // [width * height, width * height, batch]
		auto QK_scaled = ggml_scale_inplace(ctx, QK, 1.0f / float(sqrt(cin)));
		auto QK_soft_max = ggml_soft_max_inplace(ctx, QK_scaled);
		v = view_tensor(ctx, v, width * height, 1ull, cin, batch);
		auto attn_output = ggml_mul_mat(ctx, v, QK_soft_max);
		attn_output = view_tensor(ctx, attn_output, width, height, cin, batch);
		attn_output = ggml_conv_2d_s1_ph(ctx, o_proj, attn_output);
		return ggml_add_inplace(ctx, attn_output, input);
	}

private:
	ggml_tensor* norm, * norm_bias;
	ggml_tensor* q_proj, *q_bias;
	ggml_tensor* k_proj, *k_bias;
	ggml_tensor* v_proj, *v_bias;
	ggml_tensor* o_proj, *o_bias;
};

class Downsample
{
private:
	const unsigned cin;
	ggml_context* block_ctx;

public:
	Downsample(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	) : cin(ch_in)
	{
		ggml_init_params block_params = {
			.mem_size = ggml_tensor_overhead(),
			.no_alloc = true
		};
		block_ctx = ggml_init(block_params);
		conv = ggml_new_tensor_4d(block_ctx,
			GGML_TYPE_F16, 3, 3, ch_in, ch_in);
		conv_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		ggml_backend_alloc_ctx_tensors(block_ctx, backend);
		weight_data["conv"].LoadToBackend(conv);
		weight_data["conv_bias"].LoadToBackend(conv_bias);
	}

	~Downsample() {
		ggml_free(block_ctx);
	}

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input) {
		auto width = input->ne[0], height = input->ne[1];
		auto padded = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
			width + 1, height + 1, cin, input->ne[3]);
		padded = ggml_scale_inplace(ctx, padded, 0.f);
		padded = ggml_set_2d_inplace(ctx, padded, input,
			4 * (width + 1), 4 * (width + 2));
		padded = ggml_conv_2d(ctx, conv, padded, 2, 2, 1, 1, 1, 1);
		return ggml_add_inplace(ctx, padded, conv_bias);
	}

private:
	ggml_tensor* conv;
	ggml_tensor* conv_bias;
};

class Upsample
{
private:
	const unsigned cin;
	ggml_context* block_ctx;

public:
	Upsample(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	) : cin(ch_in)
	{
		ggml_init_params block_params = {
			.mem_size = ggml_tensor_overhead() * 2,
			.no_alloc = true
		};
		block_ctx = ggml_init(block_params);
		conv = ggml_new_tensor_4d(block_ctx,
			GGML_TYPE_F16, 3, 3, ch_in, ch_in);
		conv_bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F16, 1, 1, ch_in);
		ggml_backend_alloc_ctx_tensors(block_ctx, backend);
		weight_data["conv"].LoadToBackend(conv);
		weight_data["conv_bias"].LoadToBackend(conv_bias);
	}
	~Upsample() {
		ggml_free(block_ctx);
	}
	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* input) {
		auto width = input->ne[0], height = input->ne[1];
		auto x = ggml_upscale(ctx, input, 2);
		return ggml_conv_2d_s1_ph(ctx, conv, x);
	}

private:
	ggml_tensor* conv;
	ggml_tensor* conv_bias;
};

class VQ_Encoder
{
private:
	const unsigned in_channels = 3;
	const unsigned resblock_ch = 128;
	const unsigned z_channels = 256;

	ggml_context* encoder_ctx;
public:
	VQ_Encoder(
		unsigned ch_in,
		DataEater weight_data,
		ggml_backend* backend
	)
		: midres1{
			resblock_ch * ch_mult.back(),
			resblock_ch * ch_mult.back(),
			weight_data, backend
		},
		midres2{
			resblock_ch * ch_mult.back(),
			resblock_ch * ch_mult.back(),
			weight_data, backend
		},
		midattn{
			resblock_ch * ch_mult.back(),
			weight_data, backend
		}
	{
		const size_t tensor_num = 2;
		ggml_init_params block_params = {
			.mem_size = ggml_tensor_overhead() * tensor_num,
			.no_alloc = true
		};
		encoder_ctx = ggml_init(block_params);
		conv_in = ggml_new_tensor_4d(encoder_ctx, GGML_TYPE_F16,
			4, 4, ch_in, resblock_ch);
		conv_out = ggml_new_tensor_4d(encoder_ctx, GGML_TYPE_F16,
			1, 1, resblock_ch, z_channels);
		conv_in_bias = ggml_new_tensor_3d(encoder_ctx, GGML_TYPE_F16, 1, 1, resblock_ch);
		conv_out_bias = ggml_new_tensor_3d(encoder_ctx, GGML_TYPE_F16, 1, 1, z_channels);
		ggml_backend_alloc_ctx_tensors(encoder_ctx, backend);

		res_blocks.reserve(ch_mult.size());
		down_samples.reserve(ch_mult.size() - 1);
		for (size_t i = 0; i < ch_mult.size(); i++)
		{
			res_blocks.emplace_back(
				resblock_ch * (i == 0 ? 1 : ch_mult[i - 1]),
				resblock_ch * ch_mult[i],
				weight_data("conv_blocks", std::to_string(i), "res"),
				backend, 2);
			if (i != ch_mult.size() - 1)
			{
				down_samples.emplace_back(
					resblock_ch * ch_mult[i],
					weight_data("conv_blocks", std::to_string(i), "down"),
					backend);
			}
		}
	}

	ggml_tensor* compute(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = ggml_conv_2d_s1_ph(ctx, conv_in, input);
		for (auto i : std::views::iota(0ull, ch_mult.size() - 1))
		{
			x = res_blocks[i].operator()(ctx, x);
			if (i != ch_mult.size() - 1)
				x = down_samples[i].operator()(ctx, x);
		}
		x = res_blocks.back().operator()(ctx, x);
		for (auto& attn : attn_blocks)
			x = attn.operator()(ctx, x);

		x = midres1.operator()(ctx, x);
		x = midattn.operator()(ctx, x);
		x = midres2.operator()(ctx, x);

		x = ggml_group_norm_inplace(ctx, x, 32, 1e-6f);
		x = ggml_add_inplace(ctx, ggml_mul_inplace(ctx, x, norm_out), norm_out_bias);
		x = ggml_silu_inplace(ctx, x);

		return ggml_conv_2d_s1_ph(ctx, conv_out, x);
	}
private:
	ggml_tensor* conv_in;
	ggml_tensor* conv_in_bias;
	ggml_tensor* norm_out;
	ggml_tensor* norm_out_bias;
	ggml_tensor* conv_out;
	ggml_tensor* conv_out_bias;

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
	)
	{
		quant_ctx = ggml_init({
			.mem_size = ggml_tensor_overhead(),
			.no_alloc = true
		});

		embedding = ggml_new_tensor_2d(quant_ctx, GGML_TYPE_F16, 8, 16384ull);
		ggml_backend_alloc_ctx_tensors(quant_ctx, backend);

		weight_data["embedding"].LoadToBackend(embedding);
	}
	~VQ_Quantizer() {
		ggml_free(quant_ctx);
	}

	ggml_tensor* quantize(ggml_context* ctx, ggml_tensor* z)
	{
		size_t B = z->ne[3], H = z->ne[1], W = z->ne[0], C = z->ne[2];
		// z shape [B, C, H, W] -> [B, H, W, C]
		z = ggml_permute(ctx, z, 1, 2, 0, 3);
		auto z_flat = view_tensor(ctx, z, 8ull, size_t(ggml_nelements(z)) / 8);
		// L2 Norm
		auto z_norm = ggml_sum_rows(ctx, ggml_sqr(ctx, z));
		z_norm = ggml_div_inplace(ctx, z, ggml_sqrt(ctx, z_norm));
		auto z_flat_sqrsum = ggml_sum_rows(ctx, ggml_sqr(ctx, z_flat));
		auto z_flat_norm = ggml_div(ctx, z_flat, ggml_sqrt(ctx, z_flat_sqrsum));
		auto embedding_sqrsum = ggml_sum_rows(ctx, ggml_sqr(ctx, embedding));
		auto d = ggml_add_inplace(ctx, z_flat_sqrsum, embedding_sqrsum);
		auto matmul = ggml_mul_mat(ctx, embedding, z_flat);
		d = ggml_sub_inplace(ctx, d, ggml_scale_inplace(ctx, matmul, 2.f));
		auto min_idx = ggml_argmax(ctx, ggml_neg_inplace(ctx, d));
		auto z_q = ggml_get_rows(ctx, embedding, min_idx);
		z_q = ggml_permute(ctx, view_tensor(ctx, z_q, C, W, H, B), 2, 0, 1, 3);
		return view_tensor(ctx, z_q, W, H, C, B);
	}

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
	ggml_context* decoder_ctx;

public:
	VQ_Decoder(
		unsigned ch_out,
		DataEater block_data,
		ggml_backend* backend
	)
		: midres1{
			resblock_ch* ch_mult.back(),
			resblock_ch* ch_mult.back(),
			block_data("mid", "res1"), backend
		},
		midres2{
			resblock_ch * ch_mult.back(),
			resblock_ch * ch_mult.back(),
			block_data("mid", "res2"), backend
		},
		midattn{
			resblock_ch * ch_mult.back(),
			block_data("mid", "attn"), backend
		}
	{
		auto ch_mult_rev = ch_mult | std::views::reverse;

		decoder_ctx = ggml_init({
			.mem_size = ggml_tensor_overhead() * 6,
			.no_alloc = true
			});
		conv_in = ggml_new_tensor_4d(decoder_ctx, GGML_TYPE_F16,
			3, 3, z_channels, resblock_ch * ch_mult_rev[0]);
		conv_in_bias = ggml_new_tensor_3d(decoder_ctx, GGML_TYPE_F16,
			1, 1, resblock_ch * ch_mult_rev[0]);
		norm_out = ggml_new_tensor_3d(decoder_ctx, GGML_TYPE_F16, 1, 1, resblock_ch);
		norm_out_bias = ggml_new_tensor_3d(decoder_ctx, GGML_TYPE_F16, 1, 1, resblock_ch);
		conv_out = ggml_new_tensor_4d(decoder_ctx, GGML_TYPE_F16,
			3, 3, resblock_ch, 3);
		conv_out_bias = ggml_new_tensor_3d(decoder_ctx, GGML_TYPE_F16, 1, 1, 3);
		ggml_backend_alloc_ctx_tensors(decoder_ctx, backend);
		block_data["conv_in"].LoadToBackend(conv_in);
		block_data["conv_in_bias"].LoadToBackend(conv_in_bias);
		block_data["norm_out"].LoadToBackend(norm_out);
		block_data["norm_out_bias"].LoadToBackend(norm_out_bias);
		block_data["conv_out"].LoadToBackend(conv_out);
		block_data["conv_out_bias"].LoadToBackend(conv_out_bias);

		res_blocks.reserve(ch_mult.size());
		up_samples.reserve(ch_mult.size() - 1);
		attn_blocks.reserve(3);
		for (auto i : std::views::iota(0ull, ch_mult_rev.size()))
		{
			res_blocks.emplace_back(
				resblock_ch * (
					i == 0 ? ch_mult_rev.front() : ch_mult_rev[i - 1]
					),
				resblock_ch * ch_mult_rev[i],
				block_data("conv_blocks", i, "res"),
				backend, 3);
			if (i != ch_mult_rev.size() - 1)
			{
				up_samples.emplace_back(
					resblock_ch * ch_mult_rev[i],
					block_data("conv_blocks", i, "upsample"),
					backend);
			}
		}
		for (auto i : std::views::iota(0ull, 3ull))
		{
			attn_blocks.emplace_back(
				resblock_ch * ch_mult_rev[0],
				block_data("conv_blocks", 0, "attn", i),
				backend
			);
		}
	}
	~VQ_Decoder() {
		ggml_free(decoder_ctx);
	}

	ggml_tensor* operator()(ggml_context* ctx, ggml_tensor* z)
	{
		auto x = ggml_conv_2d_s1_ph(ctx, conv_in, z);
		x = ggml_add_inplace(ctx, x, conv_in_bias);

		x = midres1.operator()(ctx, x);
		x = midattn.operator()(ctx, x);
		x = midres2.operator()(ctx, x);

		x = res_blocks[0].operator()(ctx, x);
		for (auto& attn : attn_blocks)
			x = attn.operator()(ctx, x);
		x = up_samples[0].operator()(ctx, x);
		for (auto i : std::views::iota(1ull, ch_mult.size()))
		{
			x = res_blocks[i].operator()(ctx, x);
			if (i != ch_mult.size() - 1)
				x = up_samples[i].operator()(ctx, x);
		}

		x = ggml_group_norm_inplace(ctx, x, 32, 1e-6f);
		x = ggml_add_inplace(ctx, ggml_mul_inplace(ctx, x, norm_out), norm_out_bias);
		x = ggml_silu_inplace(ctx, x);

		x = ggml_conv_2d_s1_ph(ctx, conv_out, x);
		x = ggml_add_inplace(ctx, x, conv_out_bias);
		return x;
	}

private:
	ggml_tensor* conv_in;
	ggml_tensor* conv_in_bias;
	ggml_tensor* norm_out;
	ggml_tensor* norm_out_bias;
	ggml_tensor* conv_out;
	ggml_tensor* conv_out_bias;

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
	ggml_context* gen_ctx;
	ggml_backend* backend;

public:
	GenDecoder(
		ggml_backend* backend
	)
		: backend(backend)
	{
		const std::filesystem::path folder = R"(D:\Python\Janus\model-file\vq)";
		DataEater weight_data(folder);

		gen_ctx = ggml_init({
			.mem_size = ggml_tensor_overhead() * 4,
			.no_alloc = true
		});
		quant_conv = ggml_new_tensor_4d(gen_ctx, GGML_TYPE_F16, 1, 1, 256, 8);
		quant_conv_bias = ggml_new_tensor_3d(gen_ctx, GGML_TYPE_F16, 1, 1, 8);
		post_quant_conv = ggml_new_tensor_4d(gen_ctx, GGML_TYPE_F16, 1, 1, 8, 256);
		post_quant_conv_bias = ggml_new_tensor_3d(gen_ctx, GGML_TYPE_F16, 1, 1, 256);
		ggml_backend_alloc_ctx_tensors(gen_ctx, backend);

		weight_data["quant_conv"].LoadToBackend(quant_conv);
		weight_data["post_quant_conv"].LoadToBackend(post_quant_conv);

		quant = std::make_shared<VQ_Quantizer>(weight_data["quantize"], backend);
		dec = std::make_shared<VQ_Decoder>(3, weight_data["decoder"], backend);
	}

	std::vector<uint8_t> decode_img_tokens(
		std::vector<int> tokens, size_t parallel_size, size_t patch_nums)
	{
		auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

		auto ctx = ggml_init({
			.mem_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE
					  + ggml_graph_overhead(),
			.no_alloc = true
		});

		auto input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
		auto x = quant->get_rows(ctx, input);
		x = view_tensor(ctx, ggml_permute(ctx, x, 1, 0, 2, 3),
			patch_nums, patch_nums, 8ull, parallel_size);
		x = ggml_conv_2d_s1_ph(ctx, post_quant_conv, x);
		x = ggml_add_inplace(ctx, x, post_quant_conv_bias);
		x = dec->operator()(ctx, x);

		auto gr = ggml_new_graph(ctx);
		ggml_build_forward_expand(gr, x);
		ggml_gallocr_reserve(ga, gr);
		ggml_gallocr_alloc_graph(ga, gr);
		ggml_backend_tensor_set(input, tokens.data(), 0, ggml_nbytes(input));
		ggml_backend_graph_compute(backend, gr);

		std::vector<uint8_t> img_data(ggml_nbytes(x));
		ggml_backend_tensor_get(x, img_data.data(), 0, img_data.size());

		ggml_free(ctx);
		ggml_gallocr_free(ga);
		return img_data;
	}

private:
	ggml_tensor* quant_conv;
	ggml_tensor* quant_conv_bias;
	ggml_tensor* post_quant_conv;
	ggml_tensor* post_quant_conv_bias;

	std::shared_ptr<VQ_Quantizer> quant;
	std::shared_ptr<VQ_Decoder> dec;
};
