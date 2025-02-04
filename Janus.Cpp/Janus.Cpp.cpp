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
#include <exception>

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
class LlamaDecoderLayer
{
private:
	std::vector<uint8_t> graph_buffer;
	ggml_context* layer_ctx = nullptr;
	ggml_backend* layer_backend = nullptr;
public:
	LlamaDecoderLayer(ggml_type type, ggml_backend* container, bool quant = false)
	{
		layer_backend = container;
		ggml_init_params layer_param = {
			.mem_size = ggml_tensor_overhead() * 7,
			.mem_buffer = nullptr,
			.no_alloc = true
		};
		layer_ctx = ggml_init(layer_param);
		ggml_type original = type;
		if (quant) type = GGML_TYPE_Q8_K;
		q_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		k_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		v_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		o_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		gate_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 11008u);
		up_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 11008u);
		down_proj = ggml_new_tensor_2d(layer_ctx, type, 11008u, 4096u);
		ggml_set_name(q_proj, "q_proj");
		ggml_set_name(k_proj, "k_proj");
		ggml_set_name(v_proj, "v_proj");
		ggml_set_name(o_proj, "o_proj");
		ggml_set_name(gate_proj, "gate_proj");
		ggml_set_name(up_proj, "up_proj");
		ggml_set_name(down_proj, "down_proj");
		ggml_backend_alloc_ctx_tensors(layer_ctx, container);
		if (quant)
		{
			std::vector<float> original(ggml_nelements(q_proj));
			std::vector<uint8_t> quantized(ggml_nbytes(q_proj));
			ggml_quantize_chunk(type, original.data(), quantized.data(),
				0, q_proj->ne[1], q_proj->ne[0], nullptr);
			ggml_backend_tensor_set(q_proj, quantized.data(), 0, ggml_nbytes(q_proj));
		}
	}

	~LlamaDecoderLayer() {
		ggml_free(layer_ctx);
	}

	ggml_cgraph* build_llama_layer(
		size_t batch_size,
		size_t input_len,
		ggml_type type = GGML_TYPE_BF16
	)
	{
		const size_t reserved_size =
			ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
		graph_buffer.resize(reserved_size);
		ggml_init_params builder_params = {
			.mem_size = graph_buffer.size(),
			.mem_buffer = graph_buffer.data(),
			.no_alloc = true
		};
		auto ctx = ggml_init(builder_params);

		auto layer_graph = ggml_new_graph(ctx);
		auto input_emb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4096u, input_len, batch_size);
		ggml_set_name(input_emb, "input_emb");

		auto pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_len);
		auto pos_ids_data = std::views::iota(0, int32_t(input_len));
		std::copy(pos_ids_data.begin(), pos_ids_data.end(), static_cast<int32_t*>(pos_ids->data));

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
		}
		ggml_build_forward_expand(layer_graph, output);
		ggml_free(ctx);
		return layer_graph;
	}

	std::vector<uint8_t> run_layer(
		std::vector<uint8_t>& input_embs_data,
		size_t batch_size,
		size_t input_len,
		ggml_gallocr* gallocr,
		ggml_type type = GGML_TYPE_BF16
	)
	{
		auto gr = build_llama_layer(batch_size, input_len);
		ggml_gallocr_reserve(gallocr, gr);
		auto mem_size = ggml_gallocr_get_buffer_size(gallocr, 0);
		std::cout << "Layer mem usage: " << std::setprecision(2)
			<< mem_size / 1024. / 1024. << "MB" << std::endl;
		if (!ggml_gallocr_alloc_graph(gallocr, gr))
			throw std::runtime_error("Cannot allocate graph in LlamaDecoderLayer");
		auto input_embs_tensor = ggml_graph_get_tensor(gr, "input_emb");
		ggml_backend_tensor_set(input_embs_tensor, input_embs_data.data(), 0, input_embs_data.size());
		ggml_backend_graph_compute(layer_backend, gr);
		auto output = ggml_graph_node(gr, -1);
		std::vector<uint8_t> result(ggml_nbytes(output));
		ggml_backend_tensor_get(output, result.data(), 0, result.size());
		return result;
	}

private:
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
class LanguageModel
{
private:
	static constexpr int pad_id = 100015;
	ggml_backend* cuda_backend = nullptr;
	ggml_backend* blas_backend = nullptr;
	ggml_context* model_ctx = nullptr;
	const size_t gpu_offload_num;
public:
	LanguageModel(
		size_t gpu_offload_layer_num,
		ggml_type type = GGML_TYPE_BF16
	) : cuda_backend(ggml_backend_cuda_init(0)),
		blas_backend(ggml_backend_blas_init()),
		gen_head(blas_backend, type),
		gpu_offload_num(gpu_offload_layer_num)
	{
		ggml_init_params model_params = {
			.mem_size = ggml_tensor_overhead() * 1,
			.mem_buffer = nullptr,
			.no_alloc = true
		};
		model_ctx = ggml_init(model_params);

		input_embeddings = ggml_new_tensor_2d(model_ctx, type, 4096u, 102400u);
		ggml_backend_alloc_ctx_tensors(model_ctx, blas_backend);
		ggml_backend_tensor_set(input_embeddings, 0, 0, 0); // TODO: 从文件加载

		layers.reserve(30);
		for (auto i : std::views::iota(0ull, gpu_offload_layer_num))
			layers.emplace_back(type, cuda_backend);
		for (auto i : std::views::iota(gpu_offload_layer_num, 30ull))
			layers.emplace_back(type, blas_backend);
	}
	~LanguageModel() {
		ggml_free(model_ctx);
		ggml_backend_free(cuda_backend);
		ggml_backend_free(blas_backend);
	}

	std::vector<uint8_t> preprocess(
		std::vector<int> input_ids_data,
		size_t parallel_size = 16,
		size_t image_token_num_per_image = 576
	)
	{
		const size_t reserved_size =
			ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
		std::vector<uint8_t> graph_buffer(reserved_size);
		ggml_init_params builder_params = {
			.mem_size = graph_buffer.size(),
			.mem_buffer = graph_buffer.data(),
			.no_alloc = true
		};
		auto ctx = ggml_init(builder_params);

		auto pre_graph = ggml_new_graph(ctx);
		auto input_len = input_ids_data.size();
		auto input_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_len);

		// Shape: [parallel_size * 2, input_len]
		auto tokens = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, input_len, parallel_size * 2ull);
		tokens = ggml_repeat(ctx, input_ids, tokens);
		ggml_tensor* pad_id = nullptr;
		pad_id = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
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
			language_model->input_embeddings,
			ggml_view_1d(ctx, tokens, parallel_size * 2 * input_len, 0)
		);
		inputs_embeds = ggml_reshape_3d(ctx, inputs_embeds, 4096, input_len, parallel_size * 2ull);

		ggml_build_forward_expand(pre_graph, inputs_embeds);
		auto gallocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(blas_backend));
		ggml_gallocr_reserve(gallocr, pre_graph);
		auto mem_size = ggml_gallocr_get_buffer_size(gallocr, 0);
		std::cout << "Preprocess graph memory size: " << mem_size << std::endl;
		ggml_gallocr_alloc_graph(gallocr, pre_graph);
		ggml_backend_graph_compute(blas_backend, pre_graph);
		
		std::vector<uint8_t> result(ggml_nbytes(inputs_embeds));
		ggml_backend_tensor_get(inputs_embeds, result.data(), 0, result.size());
		ggml_gallocr_free(gallocr);
		ggml_free(ctx);
		return result;
	}

	std::vector<uint8_t> run_model(
		std::vector<uint8_t> input_embs_data,
		size_t parallel_size,
		size_t input_len
	)
	{
		auto cuda_ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda_backend));
		auto blas_ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(blas_backend));

		for (auto i : std::views::iota(0ull, gpu_offload_num))
		{
			auto& layer = layers[i];
			input_embs_data = layer.run_layer(input_embs_data, parallel_size * 2, input_len, cuda_ga);
		}

		return input_embs_data;
	}
private:
	ggml_tensor* input_embeddings = nullptr;
	std::vector<LlamaDecoderLayer> layers;
	struct GenHead
	{
		ggml_context* gen_head_ctx = nullptr;
		ggml_backend* gen_head_backend;
		GenHead(ggml_backend* container, ggml_type type) {
			gen_head_backend = container;
			ggml_init_params gen_head_param = {
				.mem_size = ggml_tensor_overhead() * 2,
				.mem_buffer = nullptr,
				.no_alloc = true
			};
			gen_head_ctx = ggml_init(gen_head_param);
			output_mlp_projector = ggml_new_tensor_2d(gen_head_ctx, type, 4096u, 4096u);
			vision_head = ggml_new_tensor_2d(gen_head_ctx, type, 4096u, 16384u);
			ggml_backend_alloc_ctx_tensors(gen_head_ctx, container);
		}
		~GenHead() {
			ggml_free(gen_head_ctx);
		}
		ggml_tensor* output_mlp_projector = nullptr;
		ggml_tensor* vision_head = nullptr;
	} gen_head;
} *language_model;

ggml_tensor* sample_once(ggml_tensor* probs)
{
	// auto probs32 = ggml_cast(ctx, probs, GGML_TYPE_F32);
	// auto probs_data = ggml_get_data_f32(probs32);

	return nullptr;
}

ggml_tensor* decode_image(ggml_tensor* img)
{
	return nullptr;
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
	auto input_len = inputs_embeds->ne[1];

	// 运行模型
	// Shape: [parallel_size * 2, input_len, 4096]
	ggml_tensor* output = inputs_embeds;
	auto cuda_backend = ggml_backend_cuda_init(0);
	for (auto i : std::views::iota(0, 30))
	{
		std::cout << "Layer " << std::setw(2) << i << "...";

		std::cout << std::setw(10) << "Done" << std::endl;
	}

	/*
	output = ggml_gelu_inplace(ctx,
		ggml_mul_mat(ctx, language_model->gen_head.output_mlp_projector, output));
	auto logits = ggml_mul_mat(ctx, language_model->gen_head.vision_head, output);

	// 分离条件和非条件 logits
	ggml_tensor* logit_cond = ggml_view_3d(ctx, logits, 4096, input_len / 2,
		parallel_size * 2, logits->nb[0] * 4096ull * 2, logits->nb[2], 0);
	ggml_tensor* logit_uncond = ggml_view_3d(ctx, logits, 4096, input_len / 2,
		parallel_size * 2, logits->nb[0] * 4096ull * 2, logits->nb[2], logits->nb[0] * 4096ull);

	// 计算最终 logits
	ggml_tensor* diff = ggml_sub_inplace(ctx, logit_cond, logit_uncond);
	ggml_tensor* scaled_diff = ggml_scale(ctx, diff, cfg_weight);
	ggml_tensor* final_logits = ggml_add_inplace(ctx, logit_uncond, scaled_diff);

	// 计算概率
	ggml_tensor* divided_logits = ggml_scale(ctx, final_logits, 1.f / temperature);
	ggml_tensor* probs = ggml_soft_max_inplace(ctx, divided_logits);
	*/

	return nullptr;
}

// 测试代码
void test()
{
	auto ctx_buffer = new uint8_t[8ull * 1024 * 1024 * 2];
	ggml_context* ctx1 = ggml_init(ggml_init_params{
		.mem_size = 8ull * 1024 * 1024,
		.mem_buffer = ctx_buffer,
		.no_alloc = true
		});
	ggml_context* ctx2 = ggml_init(ggml_init_params{
		.mem_size = 8ull * 1024 * 1024,
		.mem_buffer = ctx_buffer + 8ull * 1024 * 1024,
		.no_alloc = true
		});
	auto backend = ggml_backend_blas_init();
	auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

	size_t mat_shape = 2048;
	std::vector<ggml_tensor*> tensors;
	for (auto i : std::views::iota(0, 2))
		tensors.push_back(ggml_new_tensor_2d(ctx1, GGML_TYPE_F32, mat_shape, mat_shape));
	for (auto i : std::views::iota(0, 2))
		tensors.push_back(ggml_new_tensor_2d(ctx2, GGML_TYPE_F32, mat_shape, mat_shape));
	ggml_backend_alloc_ctx_tensors(ctx1, backend);
	ggml_backend_alloc_ctx_tensors(ctx2, backend);

	size_t szmat = mat_shape * mat_shape;
	std::vector<float> mat(szmat);
	for (auto i : std::views::iota(0, 4))
	{
#pragma omp parallel for
		for (int j = 0; j < szmat; j++) {
			mat[j] = rand() / 1.f / RAND_MAX;
		}
		ggml_backend_tensor_set(tensors[i], mat.data(), 0, szmat);
	}

	auto c = ggml_mul_mat(ctx1, tensors[0], tensors[1]);
	auto d = ggml_mul_mat(ctx2, tensors[2], tensors[3]);
	auto gr1 = ggml_new_graph(ctx1);
	auto gr2 = ggml_new_graph(ctx2);
	ggml_build_forward_expand(gr1, c);
	ggml_build_forward_expand(gr2, d);
	ggml_free(ctx2);
	ggml_free(ctx1);

	ggml_gallocr_reserve(ga, gr1);
	ggml_gallocr_alloc_graph(ga, gr1);
	ggml_backend_graph_compute(backend, gr1);
	ggml_backend_tensor_get(c, mat.data(), 0, szmat);

	ggml_gallocr_reserve(ga, gr2);
	ggml_gallocr_alloc_graph(ga, gr2);
	ggml_backend_graph_compute(backend, gr2);
	ggml_backend_tensor_get(d, mat.data(), 0, szmat);

	ggml_gallocr_free(ga);
}

int main(int argc, char** argv)
{
	{
		language_model = new LanguageModel(GGML_TYPE_BF16);
		std::vector<int> input(16);
		auto embeddings = language_model->preprocess(input);
		language_model->run_model(embeddings, 16, 16);
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

	const auto EleSize = ggml_type_size(GGML_TYPE_F32);

	return 0;
}
