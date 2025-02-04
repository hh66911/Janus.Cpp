// Janus.Cpp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-blas.h>
#include <ggml-cuda.h>
#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "ggml-base.lib")
#pragma comment(lib, "ggml-cpu.lib")
#pragma comment(lib, "ggml-blas.lib")
#pragma comment(lib, "ggml-cuda.lib")

#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>
#include <string>
#include <ranges>
#include <filesystem>
#include <fstream>

void inline print_shape(const ggml_tensor* tensor)
{
	std::cout << "Shape: [";
	for (auto i : std::views::iota(0, GGML_MAX_DIMS))
		std::cout << tensor->ne[i] << (i == GGML_MAX_DIMS - 1 ? "" : ", ");
	std::cout << "]\n";
}

void print_tensor_2d(const ggml_tensor* tensor)
{
	if (tensor->type == GGML_TYPE_I32)
	{
		for (auto i : std::views::iota(0, tensor->ne[1]))
		{
			for (auto j : std::views::iota(0, tensor->ne[0]))
				std::cout << static_cast<int*>(tensor->data)[i * tensor->ne[1] + j] << " ";
			std::cout << std::endl;
		}
	}
	else if (tensor->type == GGML_TYPE_F32)
	{
		for (auto i : std::views::iota(0, tensor->ne[1]))
		{
			for (auto j : std::views::iota(0, tensor->ne[0]))
				std::cout << static_cast<float*>(tensor->data)[i * tensor->ne[1] + j] << " ";
			std::cout << std::endl;
		}
	}
	else
	{
		std::cout << "Unsupported type\n";
	}
}

struct ggml_init_params params = {
	.mem_size = 20ull * 1024 * 1024 * 1024, // GB
	.mem_buffer = NULL,
	.no_alloc = false
};
ggml_context* ctx = nullptr;
ggml_context* ctx_aux = nullptr;

struct LlamaDecoderLayer
{
	LlamaDecoderLayer(ggml_type type)
	{
		q_proj = ggml_new_tensor_2d(ctx, type, 4096u, 4096u);
		k_proj = ggml_new_tensor_2d(ctx, type, 4096u, 4096u);
		v_proj = ggml_new_tensor_2d(ctx, type, 4096u, 4096u);
		o_proj = ggml_new_tensor_2d(ctx, type, 4096u, 4096u);
		gate_proj = ggml_new_tensor_2d(ctx, type, 4096u, 11008u);
		up_proj = ggml_new_tensor_2d(ctx, type, 4096u, 11008u);
		down_proj = ggml_new_tensor_2d(ctx, type, 11008u, 4096u);
		ggml_set_name(q_proj, "q_proj");
		ggml_set_name(k_proj, "k_proj");
		ggml_set_name(v_proj, "v_proj");
		ggml_set_name(o_proj, "o_proj");
		ggml_set_name(gate_proj, "gate_proj");
		ggml_set_name(up_proj, "up_proj");
		ggml_set_name(down_proj, "down_proj");
	}

	void FillGraph(ggml_cgraph* target)
	{
		*ggml_graph_get_tensor(target, "q_proj") = *q_proj;
		*ggml_graph_get_tensor(target, "k_proj") = *k_proj;
		*ggml_graph_get_tensor(target, "v_proj") = *v_proj;
		*ggml_graph_get_tensor(target, "o_proj") = *o_proj;
		*ggml_graph_get_tensor(target, "gate_proj") = *gate_proj;
		*ggml_graph_get_tensor(target, "up_proj") = *up_proj;
		*ggml_graph_get_tensor(target, "down_proj") = *down_proj;
	}

	constexpr static float eps = 1e-6f;

	// sdqa attention
	ggml_tensor* q_proj = nullptr;
	ggml_tensor* k_proj = nullptr;
	ggml_tensor* v_proj = nullptr;
	ggml_tensor* o_proj = nullptr;
	constexpr static size_t num_heads = 32;
	constexpr static size_t head_dim = 128;
	constexpr static size_t num_key_value_heads = 32;

	// mlp
	ggml_tensor* gate_proj = nullptr;
	ggml_tensor* up_proj = nullptr;
	ggml_tensor* down_proj = nullptr;
};
struct ModelData
{
	ModelData(ggml_type type = GGML_TYPE_BF16)
		: gen_head(type)
	{
		input_embeddings = ggml_new_tensor_2d(ctx, type, 4096u, 102400u);
		for (auto i : std::views::iota(0, 30))
			layers.emplace_back(type);
	}
	ggml_tensor* input_embeddings = nullptr;
	std::vector<LlamaDecoderLayer> layers;

	struct GenHead
	{
		GenHead(ggml_type type) {
			output_mlp_projector = ggml_new_tensor_2d(ctx, type, 4096u, 4096u);
			vision_head = ggml_new_tensor_2d(ctx, type, 4096u, 16384u);
		}
		ggml_tensor* output_mlp_projector = nullptr;
		ggml_tensor* vision_head = nullptr;
	} gen_head;
} *model_data;

/*
LlamaModel(
  (layers): ModuleList(
	(0-29): 30 x LlamaDecoderLayer(
	  (self_attn): LlamaSdpaAttention(
		(q_proj): Linear(in_features=4096, out_features=4096, bias=False)
		(k_proj): Linear(in_features=4096, out_features=4096, bias=False)
		(v_proj): Linear(in_features=4096, out_features=4096, bias=False)
		(o_proj): Linear(in_features=4096, out_features=4096, bias=False)
		(rotary_emb): LlamaRotaryEmbedding()
	  )
	  (mlp): LlamaMLP(
		(gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
		(up_proj): Linear(in_features=4096, out_features=11008, bias=False)
		(down_proj): Linear(in_features=11008, out_features=4096, bias=False)
		(act_fn): SiLU()
	  )
	  (input_layernorm): LlamaRMSNorm()
	  (post_attention_layernorm): LlamaRMSNorm()
	)
  )
  (norm): LlamaRMSNorm()
)*/
ggml_cgraph* build_llama_layer(
	size_t batch_size,
	size_t input_len,
	ggml_type type = GGML_TYPE_BF16
)
{
	auto layer_graph = ggml_new_graph(ctx);
	auto q_proj = ggml_new_tensor_2d(ctx_aux, type, 4096u, 4096u);
	auto k_proj = ggml_new_tensor_2d(ctx_aux, type, 4096u, 4096u);
	auto v_proj = ggml_new_tensor_2d(ctx_aux, type, 4096u, 4096u);
	auto o_proj = ggml_new_tensor_2d(ctx_aux, type, 4096u, 4096u);
	auto gate_proj = ggml_new_tensor_2d(ctx_aux, type, 4096u, 11008u);
	auto up_proj = ggml_new_tensor_2d(ctx_aux, type, 4096u, 11008u);
	auto down_proj = ggml_new_tensor_2d(ctx_aux, type, 11008u, 4096u);
	ggml_set_name(q_proj, "q_proj");
	ggml_set_name(k_proj, "k_proj");
	ggml_set_name(v_proj, "v_proj");
	ggml_set_name(o_proj, "o_proj");
	ggml_set_name(gate_proj, "gate_proj");
	ggml_set_name(up_proj, "up_proj");
	ggml_set_name(down_proj, "down_proj");
	auto input_emb = ggml_new_tensor_3d(ctx_aux, GGML_TYPE_F32, 4096u, input_len, batch_size);
	ggml_set_name(input_emb, "input_emb");

	auto pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_len);
	if (!params.no_alloc)
	{
		auto pos_ids_data = std::views::iota(0, int32_t(input_len));
		std::copy(pos_ids_data.begin(), pos_ids_data.end(), static_cast<int32_t*>(pos_ids->data));
	}

	ggml_tensor* output = nullptr;
	{
		// 层归一化
		input_emb = ggml_rms_norm(ctx, input_emb, LlamaDecoderLayer::eps);

		auto q = ggml_mul_mat(ctx, q_proj, input_emb);
		auto k = ggml_mul_mat(ctx, k_proj, input_emb);
		auto v = ggml_mul_mat(ctx, v_proj, input_emb);

		// 调整形状到 [batch, seq_len, num_head, head_dim]
		q = ggml_reshape_4d(ctx, q,
			LlamaDecoderLayer::head_dim, LlamaDecoderLayer::num_heads, input_len, batch_size);
		k = ggml_reshape_4d(ctx, k,
			LlamaDecoderLayer::head_dim, LlamaDecoderLayer::num_heads, input_len, batch_size);
		v = ggml_reshape_4d(ctx, v,
			LlamaDecoderLayer::head_dim, LlamaDecoderLayer::num_heads, input_len, batch_size);

		q = ggml_rope_inplace(ctx, q, pos_ids, LlamaDecoderLayer::head_dim, 0);
		k = ggml_rope_inplace(ctx, k, pos_ids, LlamaDecoderLayer::head_dim, 0);
		v = ggml_rope_inplace(ctx, v, pos_ids, LlamaDecoderLayer::head_dim, 0);
		q = ggml_cast(ctx, q, GGML_TYPE_BF16);
		k = ggml_cast(ctx, k, GGML_TYPE_BF16);
		v = ggml_cast(ctx, v, GGML_TYPE_BF16);

		// Shape: [batch, num_head, seq_len, head_dim]
		auto attn_output = ggml_flash_attn_ext(ctx, q, k, v, nullptr, 1.0f, 0.0f, 0.0f);

		// 调整注意力输出形状到 [batch, seq_len, num_head * head_dim]
		attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
		attn_output = ggml_reshape_3d(ctx, ggml_cont(ctx, attn_output),
			LlamaDecoderLayer::head_dim * LlamaDecoderLayer::num_heads, input_len, batch_size);
		attn_output = ggml_mul_mat(ctx, o_proj, attn_output);

		// 层归一化
		auto mlp_input = ggml_rms_norm(ctx, attn_output, LlamaDecoderLayer::eps);

		// MLP 层
		auto gate = ggml_mul_mat(ctx, gate_proj, mlp_input);
		auto up = ggml_mul_mat(ctx, up_proj, mlp_input);
		// SiLU 激活函数
		gate = ggml_silu_inplace(ctx, gate);
		// 逐元素相乘
		auto gate_up = ggml_mul_inplace(ctx, gate, up);
		// 通过 down_proj 层
		auto down = ggml_mul_mat(ctx, down_proj, gate_up);

		output = ggml_add_inplace(ctx, down, attn_output);
		ggml_set_name(output, "output");
	}
	ggml_build_forward_expand(layer_graph, output);
	return layer_graph;
}

ggml_tensor* sample_once(ggml_tensor* probs)
{
	auto probs32 = ggml_cast(ctx, probs, GGML_TYPE_F32);
	auto probs_data = ggml_get_data_f32(probs32);

	return nullptr;
}

ggml_tensor* decode_image(ggml_tensor* img)
{
	return nullptr;
}

ggml_cgraph* build_pre(
	size_t input_len,
	size_t parallel_size = 16,
	size_t image_token_num_per_image = 576
)
{
	auto pre_graph = ggml_new_graph(ctx);
	auto input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_len);
	ggml_set_name(input_ids, "input_ids");

	// Shape: [parallel_size * 2, input_len]
	auto tokens = ggml_new_tensor_2d(ctx_aux, GGML_TYPE_I32, input_len, parallel_size * 2ull);
	tokens = ggml_repeat(ctx, input_ids, tokens);
	ggml_tensor* pad_id = nullptr;
	if (params.no_alloc) {
		pad_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
	} else {
		pad_id = ggml_new_i32(ctx, 100015);
	}
	auto tokens_to_pad = ggml_view_2d(
		ctx, tokens, parallel_size, input_len - 1,
		input_len * ggml_type_size(GGML_TYPE_I32),
		(input_len + 1) * ggml_type_size(GGML_TYPE_I32)
	);
	auto repeated_pad_id = ggml_repeat(ctx, pad_id, tokens_to_pad);
	tokens_to_pad = ggml_set_2d_inplace(ctx,
		tokens_to_pad, repeated_pad_id,
		tokens_to_pad->nb[1], 0);

	// 获取输入嵌入
	ggml_tensor* inputs_embeds = ggml_get_rows(ctx,
		model_data->input_embeddings,
		ggml_view_1d(ctx, tokens, parallel_size * 2 * input_len, 0)
	);
	inputs_embeds = ggml_reshape_3d(ctx, inputs_embeds, 4096, input_len, parallel_size * 2ull);

	ggml_set_name(inputs_embeds, "input_emb");
	ggml_build_forward_expand(pre_graph, inputs_embeds);
	return pre_graph;
}

ggml_tensor* generate(
	ggml_tensor* inputs_embeds,
	float temperature = 1,
	int parallel_size = 16,
	int image_token_num_per_image = 576,
	int img_size = 384,
	int patch_size = 16,
	float cfg_weight = 5
)
{
	std::cout << "Start generating" << std::endl;
	auto gen_graph = ggml_new_graph(ctx);
	auto input_len = inputs_embeds->ne[1];

	auto pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_len);
	ggml_tensor* cfg_weight_tensor, *temperature_tensor;
	if (params.no_alloc) {
		cfg_weight_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
		temperature_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
	} else {
		cfg_weight_tensor = ggml_new_f32(ctx, cfg_weight);
		temperature_tensor = ggml_new_f32(ctx, temperature);
		auto pos_ids_data = std::views::iota(0, int32_t(input_len));
		std::copy(pos_ids_data.begin(), pos_ids_data.end(), static_cast<int*>(pos_ids->data));
	}

	// 运行模型
	// Shape: [parallel_size * 2, input_len, 4096]
	ggml_tensor* output = inputs_embeds;
	auto layer_graph = build_llama_layer(parallel_size * 2, input_len);
	auto cuda_backend = ggml_backend_cuda_init(0);
	auto backend_buffer = ggml_backend_alloc_ctx_tensors(ctx, cuda_backend);
	ggml_backend_tensor_set(
	for (auto i : std::views::iota(0, 30))
	{
		std::cout << "Layer " << std::setw(2) << i << "...";
		auto input_emb = ggml_graph_get_tensor(layer_graph, "input_emb");
		*input_emb = *ggml_view_tensor(ctx, output);
		ggml_set_name(input_emb, "input_emb");
		model_data->layers[i].FillGraph(layer_graph);
		ggml_graph_compute_with_ctx(ctx, layer_graph, 16);
		output = ggml_graph_get_tensor(layer_graph, "output");
		std::cout << std::setw(10) << "Done" << std::endl;
	}

	output = ggml_gelu_inplace(ctx,
		ggml_mul_mat(ctx, model_data->gen_head.output_mlp_projector, output));
	auto logits = ggml_mul_mat(ctx, model_data->gen_head.vision_head, output);

	// 分离条件和非条件 logits
	ggml_tensor* logit_cond = ggml_view_3d(ctx, logits, 4096, input_len / 2,
		parallel_size * 2, logits->nb[0] * 4096ull * 2, logits->nb[2], 0);
	ggml_tensor* logit_uncond = ggml_view_3d(ctx, logits, 4096, input_len / 2,
		parallel_size * 2, logits->nb[0] * 4096ull * 2, logits->nb[2], logits->nb[0] * 4096ull);

	// 计算最终 logits
	ggml_tensor* diff = ggml_sub_inplace(ctx, logit_cond, logit_uncond);
	ggml_tensor* scaled_diff = ggml_mul(ctx, diff, cfg_weight_tensor);
	ggml_tensor* final_logits = ggml_add_inplace(ctx, logit_uncond, scaled_diff);

	// 计算概率
	ggml_tensor* divided_logits = ggml_div(ctx, final_logits, temperature_tensor);
	ggml_tensor* probs = ggml_soft_max_inplace(ctx, divided_logits);

	ggml_build_forward_expand(gen_graph, probs);

	return nullptr;
}

// 测试代码
void test()
{
	int input_len = 16;
	auto pre_graph = build_pre(input_len);
	auto input_ids = ggml_graph_get_tensor(pre_graph, "input_ids");
	auto input_embeds = ggml_graph_get_tensor(pre_graph, "input_emb");
	// Fill input ids
	if (!params.no_alloc)
	{
		auto input_ids_data = std::views::iota(0, input_len);
		std::copy(input_ids_data.begin(), input_ids_data.end(), static_cast<int*>(input_ids->data));
		ggml_graph_compute_with_ctx(ctx, pre_graph, 1);
	}

	generate(input_embeds);
}

int main(int argc, char** argv)
{
	params.no_alloc = false;
	params.mem_size = params.no_alloc ? 512ull * 1024 * 1024 : 20ull * 1024 * 1024 * 1024;
	ctx = ggml_init(params);
	ctx_aux = ggml_init({ .mem_size = 4096ull, .no_alloc = true });
	{
		model_data = new ModelData(GGML_TYPE_BF16);
		test();
	}
	return -1;
	{
		// 只执行测试代码
		// model_data = new ModelData(GGML_TYPE_BF16);

		auto dataa = new float[512];
		for (auto i : std::views::iota(0, 512))
			dataa[i] = 1.f * i;
		auto b = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, 512);
		auto datab = ggml_get_data(b);

		ggml_quantize_chunk(GGML_TYPE_Q8_0, dataa, datab, 0, 1, 512, nullptr);
		delete[] dataa;

		auto c = ggml_cast(ctx, b, GGML_TYPE_F32);
		auto gr = ggml_new_graph(ctx);
		ggml_build_forward_expand(gr, c);
		ggml_graph_compute_with_ctx(ctx, gr, 1);

		dataa = ggml_get_data_f32(c);
		for (auto i : std::views::iota(0, 512))
			std::cout << dataa[i] << " ";

		ggml_free(ctx);
		ggml_free(ctx_aux);
	}
	return -1;

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <model_path>" << "<input>" << std::endl;
		return EXIT_FAILURE;
	}

	const std::filesystem::path model_path = argv[1];
	const std::filesystem::path input = argv[2];

	std::ifstream input_file(input);
	std::vector<int32_t> input_ids;
	while (input_file) {
		int32_t id;
		input_file >> id;
		input_ids.push_back(id);
	}

	ctx = ggml_init(params);
	const auto EleSize = ggml_type_size(GGML_TYPE_F32);

	ggml_free(ctx);
	ggml_free(ctx_aux);
	return 0;
}
