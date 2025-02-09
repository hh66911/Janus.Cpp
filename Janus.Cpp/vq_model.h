#pragma once

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

#include <vector>
#include <memory>

#include "tensor_utils.h"

const std::array<unsigned, 5> ch_mult{ 1, 1, 2, 2, 4 };

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
		std::span<uint8_t> weight_data,
		ggml_backend* backend
	)
		: cin(ch_in), cout(ch_out)
	{
		const size_t tensor_num = (ch_in == ch_out ? 2 : 3);
		ggml_init_params block_params = {
			.mem_size = ggml_tensor_overhead() * tensor_num,
			.no_alloc = true
		};
		block_ctx = ggml_init(block_params);

		conv_in = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			kernel_size, kernel_size, ch_in, ch_out);
		conv_out = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			kernel_size, kernel_size, ch_out, ch_out);

		if (ch_in != ch_out)
		{
			nin_shotcut = ggml_new_tensor_4d(block_ctx,
				GGML_TYPE_F16, 1, 1, ch_in, ch_out);
		}

		ggml_backend_alloc_ctx_tensors(block_ctx, backend);
	};

	~ResNetBlock() {
		ggml_free(block_ctx);
	}

	ggml_tensor* calculate(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = ggml_group_norm(ctx, input, group_norm_n_groups, 1e-6f);
		x = ggml_sigmoid_inplace(ctx, x);
		x = ggml_conv_2d_s1_ph(ctx, conv_in, x);
		x = ggml_group_norm_inplace(
			ctx, x, group_norm_n_groups, 1e-6f);
		x = ggml_sigmoid_inplace(ctx, x);
		x = ggml_conv_2d_s1_ph(ctx, conv_out, x);
		if (nin_shotcut)
			input = ggml_conv_2d_s1_ph(ctx, nin_shotcut, input);
		return ggml_add_inplace(ctx, x, input);
	}

private:
	ggml_tensor* conv_in;
	ggml_tensor* conv_out;
	ggml_tensor* nin_shotcut = nullptr;
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
		std::span<uint8_t> weight_data,
		ggml_backend* backend,
		unsigned n_blocks
	) : cin(ch_in), cout(ch_out),
		n_blocks(n_blocks)
	{
		resblocks.reserve(n_blocks);
		resblocks.emplace_back(ch_in, ch_out, weight_data, backend);
		for (size_t i = 1; i < n_blocks; i++) {
			resblocks.emplace_back(ch_out, ch_out, weight_data, backend);
		}
	}

	ggml_tensor* calculate(ggml_context* ctx, ggml_tensor* input) {
		auto x = input;
		for (auto& resblock : resblocks)
			x = resblock.calculate(ctx, x);
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
		std::span<uint8_t> weight_data,
		ggml_backend* backend
	) : cin(ch_in)
	{
		const size_t tensor_num = 4;
		ggml_init_params block_params = {
			.mem_size = ggml_tensor_overhead() * tensor_num,
			.no_alloc = true
		};
		block_ctx = ggml_init(block_params);
		q_proj = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			1, 1, ch_in, ch_in);
		k_proj = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			1, 1, ch_in, ch_in);
		v_proj = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			1, 1, ch_in, ch_in);
		o_proj = ggml_new_tensor_4d(block_ctx, GGML_TYPE_F16,
			1, 1, ch_in, ch_in);
		ggml_backend_alloc_ctx_tensors(block_ctx, backend);
	}

	~AttnBlock() {
		ggml_free(block_ctx);
	}

	ggml_tensor* calculate(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = ggml_group_norm(ctx, input, group_norm_n_groups, 1e-6f);
		auto q = ggml_conv_2d_s1_ph(ctx, q_proj, x);
		auto k = ggml_conv_2d_s1_ph(ctx, k_proj, x);
		auto v = ggml_conv_2d_s1_ph(ctx, v_proj, x);
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
	ggml_tensor* q_proj;
	ggml_tensor* k_proj;
	ggml_tensor* v_proj;
	ggml_tensor* o_proj;
};

class Downsample
{
private:
	const unsigned cin;
	ggml_context* block_ctx;

public:
	Downsample(
		unsigned ch_in,
		std::span<uint8_t> weight_data,
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
		ggml_backend_alloc_ctx_tensors(block_ctx, backend);
	}

	~Downsample() {
		ggml_free(block_ctx);
	}

	ggml_tensor* calculate(ggml_context* ctx, ggml_tensor* input) {
		auto width = input->ne[0], height = input->ne[1];
		auto padded = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
			width + 1, height + 1, cin, input->ne[3]);
		padded = ggml_scale_inplace(ctx, padded, 0.f);
		padded = ggml_set_2d_inplace(ctx, padded, input,
			4 * (width + 1), 4 * (width + 2));
		return ggml_conv_2d(ctx, conv, padded, 2, 2, 1, 1, 1, 1);
	}

private:
	ggml_tensor* conv;
};

class Upsample
{
private:
	const unsigned cin;
	ggml_context* block_ctx;

public:
	Upsample(
		unsigned ch_in,
		std::span<uint8_t> weight_data,
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
		ggml_backend_alloc_ctx_tensors(block_ctx, backend);
	}
	~Upsample() {
		ggml_free(block_ctx);
	}
	ggml_tensor* calculate(ggml_context* ctx, ggml_tensor* input) {
		auto width = input->ne[0], height = input->ne[1];
		auto x = ggml_upscale(ctx, input, 2);
		return ggml_conv_2d_s1_ph(ctx, conv, x);
	}

private:
	ggml_tensor* conv;
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
		std::span<uint8_t> weight_data,
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
		ggml_backend_alloc_ctx_tensors(encoder_ctx, backend);

		res_blocks.reserve(ch_mult.size());
		down_samples.reserve(ch_mult.size() - 1);
		for (size_t i = 0; i < ch_mult.size(); i++)
		{
			res_blocks.emplace_back(
				resblock_ch * (i == 0 ? 1 : ch_mult[i - 1]),
				resblock_ch * ch_mult[i],
				weight_data, backend, 2);
			if (i != ch_mult.size() - 1)
			{
				down_samples.emplace_back(
					resblock_ch * ch_mult[i],
					weight_data, backend);
			}
		}
	}

	ggml_tensor* compute(ggml_context* ctx, ggml_tensor* input)
	{
		auto x = ggml_conv_2d_s1_ph(ctx, conv_in, input);
		for (auto i : std::views::iota(0ull, ch_mult.size() - 1))
		{
			x = res_blocks[i].calculate(ctx, x);
			if (i != ch_mult.size() - 1)
				x = down_samples[i].calculate(ctx, x);
		}
		x = res_blocks.back().calculate(ctx, x);
		for (auto& attn : attn_blocks)
			x = attn.calculate(ctx, x);

		x = midres1.calculate(ctx, x);
		x = midattn.calculate(ctx, x);
		x = midres2.calculate(ctx, x);

		x = ggml_group_norm(ctx, x, 32, 1e-6f);
		return ggml_conv_2d_s1_ph(ctx, conv_out, x);
	}
private:
	ggml_tensor* conv_in;
	ggml_tensor* conv_out;

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
		unsigned ch_in,
		std::span<uint8_t> weight_data,
		ggml_backend* backend
	)
	{
		quant_ctx = ggml_init({
			.mem_size = ggml_tensor_overhead(),
			.no_alloc = true
		});

		embedding = ggml_new_tensor_2d(quant_ctx, GGML_TYPE_F16, 8, 16384ull);
	}
	~VQ_Quantizer() {
		ggml_free(quant_ctx);
	}

	ggml_tensor* calculate(ggml_context* ctx, ggml_tensor* z)
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
private:
	ggml_tensor* embedding;
};

class VQ_Decoder
{
private:
	const unsigned resblock_ch = 128;
	const unsigned z_channels = 256;
	ggml_context* decoder_ctx;

public:
	VQ_Decoder(
		unsigned ch_out,
		std::span<uint8_t> weight_data,
		ggml_backend* backend
	)
		: midres1{
			resblock_ch* ch_mult.back(),
			resblock_ch* ch_mult.back(),
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
		decoder_ctx = ggml_init({
			.mem_size = ggml_tensor_overhead(),
			.no_alloc = true
			});
		conv_in = ggml_new_tensor_4d(decoder_ctx, GGML_TYPE_F16,
			3, 3, z_channels, resblock_ch * ch_mult.back());
		conv_out = ggml_new_tensor_4d(decoder_ctx, GGML_TYPE_F16,
			3, 3, resblock_ch, 3);
		ggml_backend_alloc_ctx_tensors(decoder_ctx, backend);

		res_blocks.reserve(ch_mult.size());
		attn_blocks.reserve(ch_mult.size() - 1);
		for (auto i : std::views::iota(ch_mult.size(), 0ull))
		{
			res_blocks.emplace_back(
				resblock_ch * (
					i == ch_mult.size() ? ch_mult.back() : ch_mult[i]
					),
				resblock_ch * ch_mult[i - 1],
				weight_data, backend, 3);
			if (i != 1)
			{
				attn_blocks.emplace_back(
					resblock_ch * ch_mult[i - 1],
					weight_data, backend);
			}
		}
	}
	~VQ_Decoder() {
		ggml_free(decoder_ctx);
	}

	ggml_tensor* calculate(ggml_context* ctx, ggml_tensor* z)
	{
		auto x = ggml_conv_2d_s1_ph(ctx, conv_in, z);

		x = midres1.calculate(ctx, x);
		x = midattn.calculate(ctx, x);
		x = midres2.calculate(ctx, x);

		x = res_blocks[0].calculate(ctx, x);
		for (auto& attn : attn_blocks)
			x = attn.calculate(ctx, x);
		x = up_samples[0].calculate(ctx, x);
		for (auto i : std::views::iota(1ull, ch_mult.size()))
		{
			x = res_blocks[i].calculate(ctx, x);
			if (i != ch_mult.size() - 1)
				x = up_samples[i].calculate(ctx, x);
		}

		x = ggml_conv_2d_s1_ph(ctx, conv_out, x);
		return x;
	}

private:
	ggml_tensor* conv_in;
	ggml_tensor* conv_out;

	ResNetBlock midres1, midres2;
	AttnBlock midattn;

	std::vector<AttnBlock> attn_blocks;
	std::vector<GroupedResBlock> res_blocks;
	std::vector<Upsample> up_samples;
};
