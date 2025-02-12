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
private:
	std::vector<uint8_t> graph_buffer;
	ggml_context* layer_ctx = nullptr;
	ggml_backend* backend;

public:
	LlamaDecoderLayer(
		int layer_index = -1,
		ggml_backend* container = nullptr
	);

	~LlamaDecoderLayer() {
		ggml_free(layer_ctx);
	}

	void FillTo(LlamaDecoderLayer& layer, bool async = false);
	void SaveToFile(std::filesystem::path path);
	void LoadFromFile(std::filesystem::path path);
	inline static void QuantLayer(
		int layer_idx,
		std::filesystem::path dst
	) {
		LlamaDecoderLayer{ layer_idx }.SaveToFile(dst);
	}

	ggml_cgraph* build_llama_layer(
		size_t batch_size,
		size_t input_len
	);

	std::vector<uint8_t> run_layer(
		std::vector<uint8_t>& input_embs_data,
		ggml_gallocr* layer_galloc,
		size_t batch_size,
		size_t input_len,
		bool save_details = false
	);

private:
	// k v cache
	int cached_length = 0;
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
	const size_t gpu_offload_num;
	const int num_cpu_threads;
public:
	LanguageModel(
		bool load_layers,
		size_t gpu_offload_layer_num,
		int num_cpu_threads = 1
	);
	~LanguageModel() {
		ggml_free(model_ctx);
		ggml_backend_free(cuda_backend);
		ggml_backend_free(cpu_backend);
	}

	std::vector<uint8_t> preprocess(
		std::vector<int> input_ids,
		size_t parallel_size,
		size_t image_token_num_per_image
	);

	std::vector<uint8_t> postprocess(
		std::vector<uint8_t> hidden_states_data,
		size_t parallel_size,
		size_t input_len
	);

	std::vector<uint8_t> run_model(
		std::vector<uint8_t> input_embs_data,
		size_t parallel_size,
		size_t input_len
	);

	std::pair<
		std::vector<uint8_t>, std::vector<uint8_t>
	> GenHead(
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
	ggml_tensor* input_embeddings = nullptr;
	ggml_tensor* output_rms_norm = nullptr;
	std::vector<LlamaDecoderLayer> layers;
	std::vector<LlamaDecoderLayer> offloads;
	struct GenHead
	{
		ggml_context* gen_head_ctx = nullptr;
		ggml_backend* gen_head_backend;
		GenHead(ggml_backend* container);
		~GenHead() {
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
		ggml_tensor* align_embeddings = nullptr;
	} gen_head;
};