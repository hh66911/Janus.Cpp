#pragma once

#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

#include <vector>
#include <memory>
#include <functional>
#include <filesystem>
#include <random>

#include "tensor_utils.h"

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
	const int layer_idx = -1;
	constexpr static size_t max_cached_length = 2048;
	constexpr static size_t cache_incre = 32;
private:
	std::vector<uint8_t> graph_buffer;
	ggml_context* layer_ctx = nullptr;
	ggml_backend* backend;

public:
	LlamaDecoderLayer(
		int layer_index = -1,
		ggml_backend* container = nullptr
	);

	LlamaDecoderLayer(
		const LlamaDecoderLayer& other) = delete;

	LlamaDecoderLayer(LlamaDecoderLayer&& other)
		: layer_idx(other.layer_idx)
	{
		layer_ctx = other.layer_ctx;
		other.layer_ctx = nullptr;
		backend = other.backend;
		graph_buffer = std::move(other.graph_buffer);

		q_proj = other.q_proj;
		k_proj = other.k_proj;
		v_proj = other.v_proj;
		o_proj = other.o_proj;
		gate_proj = other.gate_proj;
		up_proj = other.up_proj;
		down_proj = other.down_proj;
		input_norm_weight = other.input_norm_weight;
		norm_weight = other.norm_weight;

		cached_length = other.cached_length;
		cached_k = other.cached_k;
		cached_v = other.cached_v;
		worst_case_enabled = other.worst_case_enabled;
	}

	~LlamaDecoderLayer() {
		if (layer_ctx) ggml_free(layer_ctx);
	}

	void ClearCache() {
		cached_length = 0;
		cached_k->clear();
		cached_v->clear();
	}

	void FillTo(LlamaDecoderLayer& layer, bool async = false);
	void SaveToFile(std::filesystem::path path);
	void LoadFromFile(std::filesystem::path path);
	static void QuantLayer(
		int layer_idx, ggml_backend* quant_end,
		std::filesystem::path src_folder,
		std::filesystem::path dst_folder
	);
	static LlamaDecoderLayer FromQuanted(
		int layer_idx, ggml_backend* backend,
		std::filesystem::path folder
	);

	ggml_cgraph* build_llama_layer(
		size_t batch_size,
		size_t input_len,
		bool use_cache,
		bool fast_attn
	);

	std::vector<uint8_t> run_layer(
		const std::vector<uint8_t>& input_embs_data,
		ggml_gallocr* layer_galloc,
		size_t batch_size,
		size_t input_len,
		bool save_details = false,
		bool use_cache = true
	);

	ggml_cgraph* build_refill_graph(size_t input_len, bool fast_attn);

	std::vector<uint8_t> refill_batch(
		const std::vector<uint8_t>& input_embs_data,
		ggml_gallocr* layer_galloc, size_t batch_idx
	);

private:
	// k v cache
	bool worst_case_enabled = false;
	int cached_length = 0;
	size_t cache_capacity = 0;
	std::shared_ptr<std::vector<uint8_t>> cached_k;
	std::shared_ptr<std::vector<uint8_t>> cached_v;

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
	ggml_backend* cpu_backend = nullptr;
	ggml_gallocr* cuda_ga = nullptr;
	const size_t gpu_offload_num;
	const int num_cpu_threads;
public:
	LanguageModel(
		size_t gpu_offload_layer_num,
		int num_cpu_threads = 1
	);

	LanguageModel(
		LanguageModel&& other
	) : gpu_offload_num(other.gpu_offload_num),
		num_cpu_threads(other.num_cpu_threads),
		gen_head(std::move(other.gen_head)),
		layers(std::move(other.layers)),
		offloads(std::move(other.offloads))
	{
		input_embeddings = other.input_embeddings;
		output_rms_norm = other.output_rms_norm;

		model_ctx = other.model_ctx;
		cuda_backend = other.cuda_backend;
		cuda_ga = other.cuda_ga;
		cpu_backend = other.cpu_backend;

		other.model_ctx = nullptr;
		other.cuda_backend = nullptr;
		other.cuda_ga = nullptr;
		other.cpu_backend = nullptr;
	}

	~LanguageModel() {
		if (model_ctx)
		{
			ggml_free(model_ctx);
			ggml_gallocr_free(cuda_ga);
			ggml_backend_free(cuda_backend);
			ggml_backend_free(cpu_backend);
		}
	}

	static LanguageModel LoadFromBin(
		size_t gpu_offload_layer_num,
		int num_cpu_threads,
		std::filesystem::path src_folder
	);

	std::vector<uint8_t> preprocess(
		std::vector<int> input_ids,
		size_t parallel_size,
		size_t image_token_num_per_image
	);

	std::vector<uint8_t> get_pad_embs(size_t input_len,
		bool with_sentence_start, bool with_img_start);

	std::vector<uint8_t> postprocess(
		std::vector<uint8_t> hidden_states_data,
		size_t parallel_size,
		size_t input_len
	);

	std::vector<uint8_t> run_model(
		std::vector<uint8_t> input_embs_data,
		size_t parallel_size,
		size_t input_len,
		bool dump_data = false
	);

	void refill_batch(std::vector<uint8_t> input_embs_data, size_t batch_idx);

	inline const size_t get_cached_length() const { return processed_length; }

	std::pair<
		std::vector<uint8_t>, std::vector<uint8_t>
	> run_gen_head(
		std::vector<uint8_t> outputs,
		size_t parallel_size,
		size_t input_len
	);

	std::vector<int> sample_once(
		std::vector<uint8_t> cond, std::vector<uint8_t> uncond,
		size_t parallel_size, float temperature, float cfg_weight
	);

	std::vector<uint8_t> gen_head_align(
		std::vector<int> tokens,
		size_t parallel_size
	);
private:
	size_t processed_length = 0;
	ggml_tensor* input_embeddings = nullptr;
	ggml_tensor* output_rms_norm = nullptr;
	std::vector<std::unique_ptr<LlamaDecoderLayer>> layers;
	std::vector<LlamaDecoderLayer> offloads;
	struct GenHead
	{
		ggml_context* gen_head_ctx = nullptr;
		ggml_backend* gen_head_backend = nullptr;
		ggml_gallocr* ga = nullptr;
		GenHead(ggml_backend* container);
		~GenHead() {
			ggml_gallocr_free(ga);
			ggml_free(gen_head_ctx);
		}
		std::vector<uint8_t> run_head(
			std::vector<uint8_t> hidden_states_data,
			size_t parallel_size
		);
		std::vector<uint8_t> embedding_mlp(
			std::vector<int> tokens,
			size_t parallel_size
		);
		ggml_tensor* output_mlp_projector = nullptr;
		ggml_tensor* vision_head = nullptr;
		ggml_tensor* mlp_p1 = nullptr;
		ggml_tensor* mlp_p2 = nullptr;
		ggml_tensor* mlp_p1_bias = nullptr;
		ggml_tensor* mlp_p2_bias = nullptr;
		ggml_tensor* align_embeddings = nullptr;
	} gen_head;
};