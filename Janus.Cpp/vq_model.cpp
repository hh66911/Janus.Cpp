
#include "vq_model.h"

GenDecoder::GenDecoder(ggml_backend* backend)
	: backend(backend), weight_data{ folder },
	quant_conv{ 256, quant_channels, 1, weight_data["quant_conv"], backend },
	post_quant_conv{ quant_channels, 256, 1, weight_data["post_quant_conv"], backend },
	quant{ weight_data["quantize"], backend },
	dec{ 3, weight_data["decoder"], backend }
{
}

std::vector<uint8_t> GenDecoder::decode_img_tokens(
	std::vector<int> tokens, size_t parallel_size, size_t patch_nums)

{
	if (tokens.size() != parallel_size * patch_nums * patch_nums)
		throw std::runtime_error("Tokens size mismatch");

	auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

	auto ctx = ggml_init({
		.mem_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE
				  + ggml_graph_overhead(),
		.no_alloc = true
		});
	auto gr = ggml_new_graph(ctx);

	auto input = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens.size());
	auto x = quant.get_rows(ctx, input);
	MidTensors::GetInstance().inspect_tensor(ctx, gr, x, "tokens");
	x = view_tensor(ctx, ggml_cont(ctx, ggml_transpose(ctx, x)),
		patch_nums, patch_nums, 8ull, parallel_size);
	MidTensors::GetInstance().inspect_tensor(ctx, gr, x, "tokens_permuted");
	x = post_quant_conv(ctx, x);
	MidTensors::GetInstance().inspect_tensor(ctx, gr, x, "post_quant_conv");
	x = dec(ctx, x);

	ggml_build_forward_expand(gr, x);
	ggml_gallocr_reserve(ga, gr);
	ggml_gallocr_alloc_graph(ga, gr);
	ggml_backend_tensor_set(input, tokens.data(), 0, ggml_nbytes(input));
	ggml_backend_graph_compute(backend, gr);

	MidTensors::GetInstance().SaveMidTensors("inspect/vq");

	std::vector<uint8_t> img_data(ggml_nbytes(x));
	ggml_backend_tensor_get(x, img_data.data(), 0, img_data.size());

	ggml_free(ctx);
	ggml_gallocr_free(ga);
	return img_data;
}

Convolution::Convolution(
	unsigned ch_in, unsigned ch_out,
	unsigned kernel_size,
	DataEater weight_data,
	ggml_backend* backend
)
	: cin(ch_in), cout(ch_out), kernel_size(kernel_size)
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

GroupNorm::GroupNorm(
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
	weight = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F32, 1, 1, ch_in);
	bias = ggml_new_tensor_3d(block_ctx, GGML_TYPE_F32, 1, 1, ch_in);
	ggml_backend_alloc_ctx_tensors(block_ctx, backend);
	weight_data["weight"].LoadToBackend(weight);
	weight_data["bias"].LoadToBackend(bias);
}

ggml_tensor* GroupNorm::operator()(
	ggml_context* ctx, ggml_tensor* input, bool inplace)
{
	ggml_tensor* x;
	if (inplace)
		x = ggml_group_norm_inplace(ctx, input, group_norm_n_groups, 1e-6f);
	else
		x = ggml_group_norm(ctx, input, group_norm_n_groups, 1e-6f);
	return ggml_add_inplace(ctx, ggml_mul_inplace(ctx, x, weight), bias);
}

ResNetBlock::ResNetBlock(
	unsigned ch_in, unsigned ch_out,
	DataEater weight_data,
	ggml_backend* backend
)
	: cin(ch_in), cout(ch_out), name(weight_data.GetName()),
	norm1{ ch_in, weight_data["norm1"], backend },
	conv_in{ ch_in, ch_out, 3, weight_data["conv_in"], backend },
	norm2{ ch_out, weight_data["norm2"], backend },
	conv_out{ ch_out, ch_out, 3, weight_data["conv_out"], backend }
{
	if (ch_in != ch_out)
		nin_shortcut.emplace(ch_in, ch_out, 1, weight_data["nin_shortcut"], backend);
}

ggml_tensor* ResNetBlock::operator()(ggml_context* ctx, ggml_tensor* input)
{
	auto x = norm1(ctx, input, false);
	x = swish_inplace(ctx, x);
	x = conv_in(ctx, x);
	x = norm2(ctx, x, true);
	x = swish_inplace(ctx, x);
	x = conv_out(ctx, x);
	if (nin_shortcut)
		input = (*nin_shortcut)(ctx, input);
	return ggml_add_inplace(ctx, x, input);
}

GroupedResBlock::GroupedResBlock(
	unsigned ch_in, unsigned ch_out,
	DataEater weight_data,
	ggml_backend* backend,
	unsigned n_blocks)
	: cin(ch_in), cout(ch_out),
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

AttnBlock::AttnBlock(unsigned ch_in, DataEater weight_data, ggml_backend* backend)
	: cin(ch_in),
	q_proj{ ch_in, ch_in, 1, weight_data["q_proj"], backend },
	k_proj{ ch_in, ch_in, 1, weight_data["k_proj"], backend },
	v_proj{ ch_in, ch_in, 1, weight_data["v_proj"], backend },
	o_proj{ ch_in, ch_in, 1, weight_data["o_proj"], backend },
	norm{ ch_in, weight_data["norm"], backend },
	name(weight_data.GetName())
{
}

ggml_tensor* AttnBlock::operator()(ggml_context* ctx, ggml_tensor* input)
{
	auto x = norm(ctx, input, false);
	auto q = q_proj(ctx, x);
	auto k = k_proj(ctx, x);
	auto v = v_proj(ctx, x);

	size_t width = x->ne[0], height = x->ne[1];
	size_t batch = x->ne[3];
	q = ggml_cont(ctx, ggml_permute(ctx, q, 1, 2, 0, 3));
	k = ggml_cont(ctx, ggml_permute(ctx, k, 1, 2, 0, 3));
	q = view_tensor(ctx, q, cin, width * height, batch);
	k = view_tensor(ctx, k, cin, width * height, batch);
	auto QK = ggml_mul_mat(ctx, k, q); // [width * height, width * height, batch]
	auto QK_scaled = ggml_scale_inplace(ctx, QK, 1.0f / float(sqrt(cin)));
	auto QK_soft_max = ggml_soft_max_inplace(ctx, QK_scaled);
	v = view_tensor(ctx, v, width * height, cin, batch);
	auto attn_output = ggml_mul_mat(ctx, v, QK_soft_max);
	attn_output = ggml_cont(ctx, ggml_transpose(ctx, attn_output));
	attn_output = view_tensor(ctx, attn_output, width, height, cin, batch);
	attn_output = o_proj(ctx, attn_output);
	return ggml_add_inplace(ctx, attn_output, input);
}

VQ_Encoder::VQ_Encoder(unsigned ch_in, DataEater weight_data, ggml_backend* backend)
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
	},
	conv_in{ in_channels, resblock_ch, 3, weight_data["conv_in"], backend },
	conv_out{ resblock_ch, z_channels, 3, weight_data["conv_out"], backend },
	norm_out{ resblock_ch, weight_data["norm_out"], backend }
{
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

ggml_tensor* VQ_Encoder::operator()(ggml_context* ctx, ggml_tensor* input)
{
	auto x = conv_in(ctx, input);
	for (auto i : std::views::iota(0ull, ch_mult.size() - 1))
	{
		x = res_blocks[i](ctx, x);
		if (i != ch_mult.size() - 1)
			x = down_samples[i](ctx, x);
	}
	x = res_blocks.back()(ctx, x);
	for (auto& attn : attn_blocks)
		x = attn(ctx, x);

	x = midres1(ctx, x);
	x = midattn(ctx, x);
	x = midres2(ctx, x);

	x = norm_out(ctx, x, true);
	x = swish_inplace(ctx, x);

	return conv_out(ctx, x);
}

VQ_Quantizer::VQ_Quantizer(DataEater weight_data, ggml_backend* backend)
{
	quant_ctx = ggml_init({
		.mem_size = ggml_tensor_overhead(),
		.no_alloc = true
		});

	embedding = ggml_new_tensor_2d(quant_ctx, GGML_TYPE_F16, 8, 16384ull);
	ggml_backend_alloc_ctx_tensors(quant_ctx, backend);

	weight_data["embedding"].LoadToBackend(embedding);
}

ggml_tensor* VQ_Quantizer::quantize(ggml_context* ctx, ggml_tensor* z)
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

VQ_Decoder::VQ_Decoder(unsigned ch_out, DataEater block_data, ggml_backend* backend)
	: midres1{
		resblock_ch * ch_mult.back(),
		resblock_ch * ch_mult.back(),
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
	},
	conv_in{ z_channels, resblock_ch * ch_mult.back(), 3, block_data["conv_in"], backend },
	conv_out{ resblock_ch, 3, 3, block_data["conv_out"], backend },
	norm_out{ resblock_ch, block_data["norm_out"], backend }
{
	auto ch_mult_rev = ch_mult | std::views::reverse;
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

ggml_tensor* VQ_Decoder::operator()(ggml_context* ctx, ggml_tensor* z)
{
	auto x = conv_in(ctx, z);
	MidTensors::GetInstance().inspect_tensor(ctx, nullptr, x, "conv_in");

	x = midres1(ctx, x);
	MidTensors::GetInstance().inspect_tensor(ctx, nullptr, x, "midres1");
	x = midattn(ctx, x);
	MidTensors::GetInstance().inspect_tensor(ctx, nullptr, x, "midattn");
	x = midres2(ctx, x);
	MidTensors::GetInstance().inspect_tensor(ctx, nullptr, x, "midres2");

	for (auto [res, attn] : std::views::iota(0, 3)
		| std::views::transform([this](auto i) {
			return std::make_pair(res_blocks[0][i], attn_blocks[i]);
			})) {
		x = attn(ctx, res(ctx, x));
	}

	x = up_samples[0](ctx, x);
	for (auto i : std::views::iota(1ull, ch_mult.size()))
	{
		x = res_blocks[i](ctx, x);
		MidTensors::GetInstance().inspect_tensor(
			ctx, nullptr, x, "res_blocks_" + std::to_string(i));
		if (i != ch_mult.size() - 1) {
			x = up_samples[i](ctx, x);
			MidTensors::GetInstance().inspect_tensor(
				ctx, nullptr, x, "up_samples_" + std::to_string(i));
		}
	}

	x = norm_out(ctx, x, true);
	MidTensors::GetInstance().inspect_tensor(ctx, nullptr, x, "norm_out");
	x = swish_inplace(ctx, x);
	MidTensors::GetInstance().inspect_tensor(ctx, nullptr, x, "swish_out");

	auto out = conv_out(ctx, x);
	MidTensors::GetInstance().inspect_tensor(ctx, nullptr, out, "conv_out");
	return out;
}
