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

#include "quant.h"

#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <ranges>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <functional>
#include <exception>
#include <type_traits>
#include <coroutine>

constexpr int num_threads = 1;

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

template <typename... DimTypes>
	requires (std::is_integral_v<DimTypes> && ...)
ggml_tensor* view_tensor(ggml_context* ctx, ggml_tensor* tensor, DimTypes... dims)
{
	constexpr size_t num_dims = sizeof...(DimTypes);
	static_assert(num_dims <= GGML_MAX_DIMS);
	auto type = tensor->type;
	std::array<size_t, num_dims> ne = { dims... };
	if constexpr (num_dims == 1)
		return ggml_view_1d(ctx, tensor, ne[0], 0);
	else if constexpr (num_dims == 2)
		return ggml_view_2d(ctx, tensor, ne[0], ne[1],
			ggml_row_size(type, ne[0]), 0);
	else if constexpr (num_dims == 3)
		return ggml_view_3d(ctx, tensor, ne[0], ne[1], ne[2],
			ggml_row_size(type, ne[0]),
			ggml_row_size(type, ne[0] * ne[1]),
			0);
	else if constexpr (num_dims == 4)
		return ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3],
			ggml_row_size(type, ne[0]),
			ggml_row_size(type, ne[0] * ne[1]),
			ggml_row_size(type, ne[0] * ne[1] * ne[2]),
			0);
	else
		static_assert(false);
}

ggml_tensor* flatten_tensor(ggml_context* ctx, ggml_tensor* tensor)
{
	auto ne = ggml_nelements(tensor);
	return ggml_view_1d(ctx, tensor, ne, 0);
}

class ModelTimer
{
public:
	enum class TimerType
	{
		Model,
		Layer,
		CopyTensor,
		Compute,
		BuildGraph,
		__Count
	};
private:
	std::array<
		std::chrono::time_point<std::chrono::high_resolution_clock>,
		static_cast<size_t>(TimerType::__Count)
	> start_time_points;
	std::array<std::chrono::milliseconds,
		static_cast<size_t>(TimerType::__Count)> durations;
public:
	void Start(TimerType type) {
		start_time_points[static_cast<size_t>(type)] = std::chrono::high_resolution_clock::now();
	}
	void Stop(TimerType type) {
		auto end_time_point = std::chrono::high_resolution_clock::now();
		durations[static_cast<size_t>(type)] +=
			std::chrono::duration_cast<std::chrono::milliseconds>(
			end_time_point - start_time_points[static_cast<size_t>(type)]);
	}
	std::chrono::milliseconds GetDuration(TimerType type) const {
		return durations[static_cast<size_t>(type)];
	}
	void PrintTimeConsumed(TimerType type) const {
		std::cout << "Time consumed for ";
		switch (type)
		{
		case TimerType::Model:
			std::cout << "model";
			break;
		case TimerType::Layer:
			std::cout << "layer";
			break;
		case TimerType::CopyTensor:
			std::cout << "copying tensor";
			break;
		case TimerType::Compute:
			std::cout << "computing";
			break;
		case TimerType::BuildGraph:
			std::cout << "building graph";
			break;
		default:
			throw std::invalid_argument("Invalid TimerType");
		}
		std::cout << ": " << durations[static_cast<size_t>(type)].count() << "ms\n";
	}
	void PrintTimeConsumedAll() const {
		for (size_t i = 0; i < static_cast<size_t>(TimerType::__Count); ++i)
			PrintTimeConsumed(static_cast<TimerType>(i));
	}
} global_timer;

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
public:
	constexpr static float eps = 1e-6f;
	constexpr static size_t num_heads = 32;
	constexpr static size_t head_dim = 128;
	constexpr static size_t num_key_value_heads = 32;
private:
	std::vector<uint8_t> graph_buffer;
	ggml_context* layer_ctx = nullptr;
	ggml_backend* backend;

	std::vector<ggml_tensor*> get_mid_tensors(ggml_cgraph* gr)
	{
		std::vector<ggml_tensor*> mid_tensors;
		std::vector<std::string> mid_tensor_names = {
			"rms_normed_input",
			"q", "k", "v",
			"q_pos", "k_pos", "v_pos",
			"attn_out+input",
			"mlp_input",
			"residual_output"
		};
		for (const auto& name : mid_tensor_names)
		{
			auto tensor = ggml_graph_get_tensor(gr, name.c_str());
			if (tensor)
				mid_tensors.push_back(tensor);
		}
		return mid_tensors;
	}
public:
	LlamaDecoderLayer(
		int layer_idx = -1,
		ggml_backend* container = nullptr
	) : backend(container)
	{
		ggml_type type = GGML_TYPE_Q8_0;
		constexpr size_t num_tensors = 9;
		if (backend)
		{
			ggml_init_params layer_param = {
				.mem_size = ggml_tensor_overhead() * num_tensors,
				.mem_buffer = nullptr,
				.no_alloc = true
			};
			layer_ctx = ggml_init(layer_param);
		}
		else
		{
			constexpr size_t num_elements =
				4096ull * 4096 * 4 + 4096ull * 11008 * 3;
			ggml_init_params layer_param = {
				// ~400 MB when type = BF16
				.mem_size = num_elements * ggml_type_size(type) / ggml_blck_size(type)
						  + 4096 * ggml_type_size(GGML_TYPE_F32) * 2
						  + num_tensors * ggml_tensor_overhead(),
			};
			layer_ctx = ggml_init(layer_param);
		}
		input_norm_weight = ggml_new_tensor_1d(layer_ctx, GGML_TYPE_F32, 4096u);
		q_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		k_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		v_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		o_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		gate_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 11008u);
		up_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 11008u);
		down_proj = ggml_new_tensor_2d(layer_ctx, type, 11008u, 4096u);
		norm_weight = ggml_new_tensor_1d(layer_ctx, GGML_TYPE_F32, 4096u);
		ggml_set_name(q_proj, "q_proj");
		ggml_set_name(k_proj, "k_proj");
		ggml_set_name(v_proj, "v_proj");
		ggml_set_name(o_proj, "o_proj");
		ggml_set_name(gate_proj, "gate_proj");
		ggml_set_name(up_proj, "up_proj");
		ggml_set_name(down_proj, "down_proj");
		if (backend)
			ggml_backend_alloc_ctx_tensors(layer_ctx, backend);
		else if (layer_idx >= 0)
		{
			// 从文件加载权重并量化
			F32TensorFromFile(layer_ctx, norm_weight,
				GetWeightFileName(layer_idx, "post_attention_layernorm"));
			F32TensorFromFile(layer_ctx, input_norm_weight,
				GetWeightFileName(layer_idx, "input_layernorm"));
			QuantTensorFromFile(layer_ctx, q_proj, GetWeightFileName(layer_idx, "self_attn.q_proj"));
			QuantTensorFromFile(layer_ctx, k_proj, GetWeightFileName(layer_idx, "self_attn.k_proj"));
			QuantTensorFromFile(layer_ctx, v_proj, GetWeightFileName(layer_idx, "self_attn.v_proj"));
			QuantTensorFromFile(layer_ctx, o_proj, GetWeightFileName(layer_idx, "self_attn.o_proj"));
			QuantTensorFromFile(layer_ctx, gate_proj, GetWeightFileName(layer_idx, "mlp.gate_proj"));
			QuantTensorFromFile(layer_ctx, up_proj, GetWeightFileName(layer_idx, "mlp.up_proj"));
			QuantTensorFromFile(layer_ctx, down_proj, GetWeightFileName(layer_idx, "mlp.down_proj"));
		}
	}

	~LlamaDecoderLayer() {
		ggml_free(layer_ctx);
	}

	void FillTo(LlamaDecoderLayer& layer, bool async = false)
	{
		std::function<void(ggml_tensor*, const void*, size_t, size_t)> set_tensor;
		if (async)
			set_tensor = std::bind(
				ggml_backend_tensor_set_async, backend,
				std::placeholders::_1, std::placeholders::_2,
				std::placeholders::_3, std::placeholders::_4);
		else
			set_tensor = ggml_backend_tensor_set;
		set_tensor(layer.input_norm_weight, input_norm_weight->data, 0, ggml_nbytes(input_norm_weight));
		set_tensor(layer.q_proj, q_proj->data, 0, ggml_nbytes(q_proj));
		set_tensor(layer.k_proj, k_proj->data, 0, ggml_nbytes(k_proj));
		set_tensor(layer.v_proj, v_proj->data, 0, ggml_nbytes(v_proj));
		set_tensor(layer.o_proj, o_proj->data, 0, ggml_nbytes(o_proj));
		set_tensor(layer.gate_proj, gate_proj->data, 0, ggml_nbytes(gate_proj));
		set_tensor(layer.up_proj, up_proj->data, 0, ggml_nbytes(up_proj));
		set_tensor(layer.down_proj, down_proj->data, 0, ggml_nbytes(down_proj));
		set_tensor(layer.norm_weight, norm_weight->data, 0, ggml_nbytes(norm_weight));
	}

	ggml_cgraph* build_llama_layer(
		size_t batch_size,
		size_t input_len
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
		ggml_set_name(pos_ids, "pos_ids");

		ggml_tensor* output = nullptr;
		{
			// 层归一化
			auto rms_normed_input = ggml_rms_norm(ctx, input_emb, eps);
			rms_normed_input = ggml_mul_inplace(ctx, rms_normed_input, input_norm_weight);
			ggml_set_name(rms_normed_input, "rms_normed_input");

			auto q = ggml_mul_mat(ctx, q_proj, rms_normed_input);
			auto k = ggml_mul_mat(ctx, k_proj, rms_normed_input);
			auto v = ggml_mul_mat(ctx, v_proj, rms_normed_input);

			// 调整形状到 [batch, seq_len, num_head, head_dim]
			q = view_tensor(ctx, q,
				head_dim, num_heads, input_len, batch_size);
			k = view_tensor(ctx, k,
				head_dim, num_heads, input_len, batch_size);
			v = view_tensor(ctx, v,
				head_dim, num_heads, input_len, batch_size);
			ggml_set_name(q, "q");
			ggml_set_name(k, "k");
			ggml_set_name(v, "v");

			q = ggml_rope_inplace(ctx, q, pos_ids, head_dim, 0);
			k = ggml_rope_inplace(ctx, k, pos_ids, head_dim, 0);
			v = ggml_rope_inplace(ctx, v, pos_ids, head_dim, 0);
			k = view_tensor(ctx, ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)),
				head_dim, input_len, num_heads, batch_size);
			q = view_tensor(ctx, ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)),
				head_dim, input_len, num_heads, batch_size);
			v = view_tensor(ctx, ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3)),
				input_len, head_dim, num_heads, batch_size);
			ggml_set_name(q, "q_pos");
			ggml_set_name(k, "k_pos");
			ggml_set_name(v, "v_pos");

			// K * Q Shape: [batch, num_head * head_dim, seq_len, seq_len]
			ggml_tensor* KQ = ggml_mul_mat(ctx, k, q); // 800ms !!!!!!!!!!!!!!!!
			// KQ_scaled = KQ / sqrt(n_embd/n_head)
			ggml_tensor* KQ_scaled = ggml_scale_inplace(ctx, KQ,
				1.0f / sqrt(float(head_dim) / num_heads));
			// KQ_masked = mask_past(KQ_scaled)
			ggml_tensor* KQ_masked = ggml_diag_mask_inf_inplace(ctx, KQ_scaled,
				static_cast<int>(input_len));
			// KQ = soft_max(KQ_masked)
			ggml_tensor* KQ_soft_max = ggml_soft_max_inplace(ctx, KQ_masked);
			// KQV = transpose(V) * KQ_soft_max 800ms !!!!!!!!!!!!!!!!
			auto attn_output = ggml_permute(ctx, ggml_mul_mat(ctx, KQ_soft_max, v), 2, 0, 3, 1); 
			attn_output = view_tensor(ctx, ggml_cont(ctx, attn_output),
				head_dim * num_heads, input_len, batch_size);
			attn_output = ggml_mul_mat(ctx, o_proj, attn_output);
			auto residual = ggml_add_inplace(ctx,
				flatten_tensor(ctx, attn_output), flatten_tensor(ctx, input_emb));
			ggml_set_name(residual, "attn_out+input");

			// 层归一化
			auto mlp_input = ggml_rms_norm(ctx, attn_output, eps);
			mlp_input = ggml_mul_inplace(ctx, mlp_input, norm_weight);
			ggml_set_name(mlp_input, "mlp_input");

			// MLP 层
			auto gate = ggml_mul_mat(ctx, gate_proj, mlp_input);
			auto up = ggml_mul_mat(ctx, up_proj, mlp_input);
			// SiLU 激活函数
			gate = ggml_silu_inplace(ctx, gate);
			// 逐元素相乘
			auto gate_up = ggml_mul_inplace(ctx, gate, up);
			// 通过 down_proj 层
			auto down = ggml_mul_mat(ctx, down_proj, gate_up);

			output = ggml_add_inplace(ctx,
				flatten_tensor(ctx, down), flatten_tensor(ctx, residual));
			ggml_set_name(output, "residual_output");
		}
		ggml_build_forward_expand(layer_graph, output);
		ggml_free(ctx);
		return layer_graph;
	}

	std::vector<uint8_t> run_layer(
		std::vector<uint8_t>& input_embs_data,
		ggml_gallocr* layer_galloc,
		size_t batch_size,
		size_t input_len,
		bool save_mid_tensors = false
	)
	{
		global_timer.Start(ModelTimer::TimerType::Layer);
		if (!backend)
		{
			auto gr = build_llama_layer(batch_size, input_len);
			backend = ggml_backend_cpu_init();
			ggml_backend_cpu_set_n_threads(backend, num_threads);
			auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
			ggml_gallocr_reserve(ga, gr);
			if (!ggml_gallocr_alloc_graph(ga, gr))
				throw std::runtime_error("Cannot allocate graph in LlamaDecoderLayer");

			auto input_embs_tensor = ggml_graph_get_tensor(gr, "input_emb");
			ggml_backend_tensor_set(input_embs_tensor, input_embs_data.data(), 0, input_embs_data.size());
			auto pos_ids = ggml_graph_get_tensor(gr, "pos_ids");
			auto pos_ids_generator = std::views::iota(0, int32_t(input_len));
			std::vector<int> pos_ids_data(input_len);
			std::copy(pos_ids_generator.begin(), pos_ids_generator.end(), pos_ids_data.begin());
			ggml_backend_tensor_set(pos_ids, pos_ids_data.data(), 0, ggml_nbytes(pos_ids));

			ggml_backend_graph_compute(backend, gr);
			auto output = ggml_graph_node(gr, -1);
			std::vector<uint8_t> result(ggml_nbytes(output));
			ggml_backend_tensor_get(output, result.data(), 0, result.size());
			global_timer.Stop(ModelTimer::TimerType::Layer);

			if (save_mid_tensors)
			{
				auto mid_tensors = get_mid_tensors(gr);
				for (auto tensor : mid_tensors)
				{
					std::ofstream ofs(
						"inspect/" + std::string(tensor->name) + ".bin", std::ios::binary);
					std::vector<char> data(ggml_nbytes(tensor));
					ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
					ofs.write(data.data(), data.size());
				}
			}

			ggml_gallocr_free(ga);
			ggml_backend_free(backend);
			backend = nullptr;
			return result;
		}

		global_timer.Start(ModelTimer::TimerType::BuildGraph);
		auto gr = build_llama_layer(batch_size, input_len);
		ggml_gallocr_reserve(layer_galloc, gr);
		if (!ggml_gallocr_alloc_graph(layer_galloc, gr))
			throw std::runtime_error("Cannot allocate graph in LlamaDecoderLayer");
		global_timer.Stop(ModelTimer::TimerType::BuildGraph);

		global_timer.Start(ModelTimer::TimerType::CopyTensor);
		auto input_embs_tensor = ggml_graph_get_tensor(gr, "input_emb");
		ggml_backend_tensor_set(input_embs_tensor, input_embs_data.data(), 0, input_embs_data.size());
		auto pos_ids = ggml_graph_get_tensor(gr, "pos_ids");
		auto pos_ids_generator = std::views::iota(0, int32_t(input_len));
		std::vector<int> pos_ids_data(input_len);
		std::copy(pos_ids_generator.begin(), pos_ids_generator.end(), pos_ids_data.begin());
		ggml_backend_tensor_set(pos_ids, pos_ids_data.data(), 0, ggml_nbytes(pos_ids));
		global_timer.Stop(ModelTimer::TimerType::CopyTensor);

		global_timer.Start(ModelTimer::TimerType::Compute);
		ggml_backend_graph_compute(backend, gr);
		global_timer.Stop(ModelTimer::TimerType::Compute);

		global_timer.Start(ModelTimer::TimerType::CopyTensor);
		auto output = ggml_graph_node(gr, -1);
		std::vector<uint8_t> result(ggml_nbytes(output));
		ggml_backend_tensor_get(output, result.data(), 0, result.size());
		global_timer.Stop(ModelTimer::TimerType::CopyTensor);

		global_timer.Stop(ModelTimer::TimerType::Layer);

		if (save_mid_tensors)
		{
			auto mid_tensors = get_mid_tensors(gr);
			for (auto tensor : mid_tensors)
			{
				std::ofstream ofs(
					"inspect/cuda/" + std::string(tensor->name) + ".bin", std::ios::binary);
				std::vector<char> data(ggml_nbytes(tensor));
				ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
				ofs.write(data.data(), data.size());
			}
		}

		return result;
	}

private:
	// sdqa attention
	ggml_tensor* q_proj = nullptr;
	ggml_tensor* k_proj = nullptr;
	ggml_tensor* v_proj = nullptr;
	ggml_tensor* o_proj = nullptr;

	// mlp
	ggml_tensor* gate_proj = nullptr;
	ggml_tensor* up_proj = nullptr;
	ggml_tensor* down_proj = nullptr;

	// layer norm
	ggml_tensor* input_norm_weight = nullptr;
	ggml_tensor* norm_weight = nullptr;
};

class LanguageModel
{
private:
	static constexpr int pad_id = 100015;
	ggml_context* model_ctx = nullptr;
	ggml_backend* cuda_backend = nullptr;
	ggml_backend* blas_backend = nullptr;
	const size_t gpu_offload_num;
public:
	LanguageModel(
		bool load_layers,
		size_t gpu_offload_layer_num
	) : cuda_backend(ggml_backend_cuda_init(0)),
		blas_backend(ggml_backend_blas_init()),
		gen_head(blas_backend, GGML_TYPE_F16),
		gpu_offload_num(gpu_offload_layer_num)
	{
		ggml_init_params model_params = {
			.mem_size = ggml_tensor_overhead() * 2 +
				4096ull * 102400 * ggml_type_size(GGML_TYPE_F16) +
				4096ull * ggml_type_size(GGML_TYPE_F32),
			.mem_buffer = nullptr,
			.no_alloc = false
		};
		model_ctx = ggml_init(model_params);

		// 从文件加载
		input_embeddings = ggml_new_tensor_2d(
			model_ctx, GGML_TYPE_F16, 4096u, 102400u);
		F16TensorFromFile(model_ctx, input_embeddings,
			R"(D:\Python\Janus\model-file\embed_tokens.bin)");
		output_rms_norm = ggml_new_tensor_1d(model_ctx, GGML_TYPE_F32, 4096ull);
		F32TensorFromFile(model_ctx, output_rms_norm,
			R"(D:\Python\Janus\model-file\norm.weight.bin)");

		if (!load_layers)
			return;
		layers.reserve(30);
		for (auto i : std::views::iota(0, 30))
			layers.emplace_back(i);

		offloads.reserve(gpu_offload_num);
		for (auto i : std::views::iota(0ull, gpu_offload_num))
		{
			offloads.emplace_back(-1, cuda_backend);
			layers[i].FillTo(offloads[i]);
		}
	}
	~LanguageModel() {
		ggml_free(model_ctx);
		ggml_backend_free(cuda_backend);
		ggml_backend_free(blas_backend);
	}

	std::vector<uint8_t> preprocess(
		std::vector<int> input_ids,
		size_t parallel_size = 16,
		size_t image_token_num_per_image = 576
	)
	{
		auto input_len = input_ids.size();

		// Shape: [parallel_size * 2, input_len]
		std::vector<int> tokens_data(input_len * parallel_size * 2ull);
#pragma omp parallel for
		for (int i = 0; i < int(parallel_size) * 2; i++)
		{
			if (i % 2 == 0)
				std::copy(input_ids.begin(), input_ids.end(), tokens_data.begin() + i * input_len);
			else
			{
				tokens_data[i * input_len] = input_ids[0];
				std::fill(tokens_data.begin() + i * input_len + 1,
					tokens_data.begin() + (i + 1) * input_len, pad_id);
			}
		}

		ggml_init_params builder_params = {
			.mem_size = (tokens_data.size() * 4097) * 4 +
				ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
		};
		auto ctx = ggml_init(builder_params);
		auto pre_graph = ggml_new_graph(ctx);

		auto tokens = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, input_len, parallel_size * 2ull);
		std::copy(tokens_data.begin(), tokens_data.end(), static_cast<int*>(tokens->data));
		// 获取输入嵌入
		ggml_tensor* inputs_embeds = ggml_get_rows(ctx,
			input_embeddings,
			ggml_view_1d(ctx, tokens, parallel_size * 2 * input_len, 0)
		);
		inputs_embeds = view_tensor(ctx, inputs_embeds, 4096ull, input_len, parallel_size * 2ull);

		ggml_build_forward_expand(pre_graph, inputs_embeds);
		ggml_graph_compute_with_ctx(ctx, pre_graph, num_threads);
		
		std::vector<uint8_t> result(ggml_nbytes(inputs_embeds));
		memcpy(result.data(), inputs_embeds->data, result.size());
		ggml_free(ctx);
		return result;
	}

	std::vector<uint8_t> postprocess(
		std::vector<uint8_t> hidden_states_data,
		size_t parallel_size,
		size_t input_len
	)
	{
		ggml_init_params builder_params = {
			.mem_size = hidden_states_data.size() +
				ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead()
		};
		auto ctx = ggml_init(builder_params);
		auto post_graph = ggml_new_graph(ctx);
		auto hidden_states = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
			4096ull, input_len, parallel_size * 2);
		auto out_states = ggml_rms_norm_inplace(ctx, hidden_states, LlamaDecoderLayer::eps);
		out_states = ggml_mul_inplace(ctx, hidden_states, output_rms_norm);
		ggml_build_forward_expand(post_graph, out_states);
		std::copy(hidden_states_data.begin(), hidden_states_data.end(),
			static_cast<uint8_t*>(hidden_states->data));
		ggml_graph_compute_with_ctx(ctx, post_graph, num_threads);
		std::vector<uint8_t> result(ggml_nbytes(out_states));
		memcpy(result.data(), out_states->data, result.size());
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
		global_timer.Start(ModelTimer::TimerType::Model);
		for (auto i : std::views::iota(0ull, gpu_offload_num))
		{
			// 运行模型
			auto& layer = offloads[i];
			input_embs_data = layer.run_layer(
				input_embs_data, cuda_ga, parallel_size * 2, input_len);
		}
		global_timer.Stop(ModelTimer::TimerType::Model);
		global_timer.PrintTimeConsumedAll();

		input_embs_data = postprocess(input_embs_data, parallel_size, input_len);

		return input_embs_data;
	}
private:
	ggml_tensor* input_embeddings = nullptr;
	ggml_tensor* output_rms_norm = nullptr;
	std::vector<LlamaDecoderLayer> layers;
	std::vector<LlamaDecoderLayer> offloads;
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
	std::vector<uint8_t> emb(8192ull * 1024);
	std::ifstream emb_file("inspect/embeddings.bin", std::ios::binary);
	if (!emb_file.is_open())
		throw std::runtime_error("Failed to open file: embeddings.bin");
	emb_file.read(reinterpret_cast<char*>(emb.data()), emb.size());
	emb_file.close();
	LlamaDecoderLayer layer{ 0 };
	auto result = layer.run_layer(emb, nullptr, 32, 16, true);
	/*
	auto cuda = ggml_backend_cuda_init(0);
	auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda));
	LlamaDecoderLayer gpu_layer{ -1, cuda };
	layer.FillTo(gpu_layer);
	auto gpu_result = gpu_layer.run_layer(emb, ga, 32, 16, true);
	ggml_gallocr_free(ga);
	ggml_backend_free(cuda);
	*/
}

int main(int argc, char** argv)
{
	test(); return -1;
	{
		language_model = new LanguageModel(true, 30);
		std::vector<int> input(16);
		for (auto i : std::views::iota(1, 16))
			input[i] = i;
		input[0] = 100000;
		auto embeddings = language_model->preprocess(input);
		std::ofstream emb_file("inspect/embeddings.bin", std::ios::binary);
		emb_file.write(reinterpret_cast<const char*>(embeddings.data()), embeddings.size());
		auto result = language_model->run_model(embeddings, 16, 16);
		delete language_model;
		std::cout << "Writing result to file\n";
		std::ofstream file("inspect/result.bin", std::ios::binary);
		if (!file.is_open())
			throw std::runtime_error("Failed to open file: result.bin");
		file.write(reinterpret_cast<const char*>(result.data()), result.size());
		return 0;
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
