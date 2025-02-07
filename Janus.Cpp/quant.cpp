#include "quant.h"

#include <iostream>

void QuantTensorFromFile(ggml_context* ctx,
	ggml_tensor* dst, std::filesystem::path path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Failed to open file: " + path.string());
	std::vector<ggml_fp16_t> buffer(ggml_nelements(dst));
	file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(ggml_fp16_t));
	std::vector<float> float_buffer(buffer.size());
	if (buffer.size() > std::numeric_limits<int>::max())
		throw std::runtime_error("Too many elements");
#pragma omp parallel for
	for (int i = 0; i < buffer.size(); ++i)
		float_buffer[i] = ggml_fp16_to_fp32(buffer[i]);
	buffer.clear();
	ggml_quantize_chunk(dst->type, float_buffer.data(),
		dst->data, 0, dst->ne[1], dst->ne[0], nullptr);
}

void F32TensorFromFile(ggml_context* ctx, ggml_tensor* dst, std::filesystem::path path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Failed to open file: " + path.string());
	std::vector<ggml_fp16_t> buffer(ggml_nelements(dst));
	file.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(ggml_fp16_t));
#pragma omp parallel for
	for (int i = 0; i < buffer.size(); ++i)
		static_cast<float*>(dst->data)[i] = ggml_fp16_to_fp32(buffer[i]);
}

void F16TensorFromFile(ggml_context* ctx, ggml_tensor* dst, std::filesystem::path path)
{
	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Failed to open file: " + path.string());
	file.read(reinterpret_cast<char*>(dst->data), ggml_nbytes(dst));
}

void ConvertModelFile(std::filesystem::path src, std::filesystem::path dst)
{
	auto type = GGML_TYPE_Q8_0;
	constexpr size_t num_tensors = 9;
	constexpr size_t num_elements =
		4096ull * 4096 * 4 + 4096ull * 11008 * 3 +
		4096ull * 2 * sizeof(float);
	std::vector<uint8_t> buffer(
		num_elements * ggml_type_size(type) / ggml_blck_size(type)
		+ num_tensors * ggml_tensor_overhead());
	ggml_init_params layer_param = {
		// ~400 MB when type = BF16
		.mem_size = num_elements * ggml_type_size(type) / ggml_blck_size(type)
				  + num_tensors * ggml_tensor_overhead(),
		.mem_buffer = buffer.data(),
	};

	for (auto layer_idx : std::views::iota(0, 30))
	{
		auto gguf = gguf_init_empty();

		auto layer_ctx = ggml_init(layer_param);
		auto input_norm_weight = ggml_new_tensor_1d(layer_ctx, GGML_TYPE_F32, 4096u);
		auto q_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		auto k_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		auto v_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		auto o_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 4096u);
		auto gate_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 11008u);
		auto up_proj = ggml_new_tensor_2d(layer_ctx, type, 4096u, 11008u);
		auto down_proj = ggml_new_tensor_2d(layer_ctx, type, 11008u, 4096u);
		auto norm_weight = ggml_new_tensor_1d(layer_ctx, GGML_TYPE_F32, 4096u);
		QuantTensorFromFile(layer_ctx, q_proj, GetWeightFileName(layer_idx, "self_attn.q_proj"));
		QuantTensorFromFile(layer_ctx, k_proj, GetWeightFileName(layer_idx, "self_attn.k_proj"));
		QuantTensorFromFile(layer_ctx, v_proj, GetWeightFileName(layer_idx, "self_attn.v_proj"));
		QuantTensorFromFile(layer_ctx, o_proj, GetWeightFileName(layer_idx, "self_attn.o_proj"));
		QuantTensorFromFile(layer_ctx, gate_proj, GetWeightFileName(layer_idx, "mlp.gate_proj"));
		QuantTensorFromFile(layer_ctx, up_proj, GetWeightFileName(layer_idx, "mlp.up_proj"));
		QuantTensorFromFile(layer_ctx, down_proj, GetWeightFileName(layer_idx, "mlp.down_proj"));
		F32TensorFromFile(layer_ctx, norm_weight,
			GetWeightFileName(layer_idx, "post_attention_layernorm"));
		F32TensorFromFile(layer_ctx, input_norm_weight,
			GetWeightFileName(layer_idx, "input_layernorm"));

		ggml_set_name(input_norm_weight,
			GetLayerWeightName(layer_idx, "input_layernorm").c_str());
		ggml_set_name(q_proj, GetLayerWeightName(layer_idx, "self_attn.q_proj").c_str());
		ggml_set_name(k_proj, GetLayerWeightName(layer_idx, "self_attn.k_proj").c_str());
		ggml_set_name(v_proj, GetLayerWeightName(layer_idx, "self_attn.v_proj").c_str());
		ggml_set_name(o_proj, GetLayerWeightName(layer_idx, "self_attn.o_proj").c_str());
		ggml_set_name(gate_proj, GetLayerWeightName(layer_idx, "mlp.gate_proj").c_str());
		ggml_set_name(up_proj, GetLayerWeightName(layer_idx, "mlp.up_proj").c_str());
		ggml_set_name(down_proj, GetLayerWeightName(layer_idx, "mlp.down_proj").c_str());
		ggml_set_name(norm_weight,
			GetLayerWeightName(layer_idx, "post_attention_layernorm").c_str());

		gguf_add_tensor(gguf, input_norm_weight);
		gguf_add_tensor(gguf, q_proj);
		gguf_add_tensor(gguf, k_proj);
		gguf_add_tensor(gguf, v_proj);
		gguf_add_tensor(gguf, o_proj);
		gguf_add_tensor(gguf, gate_proj);
		gguf_add_tensor(gguf, up_proj);
		gguf_add_tensor(gguf, down_proj);
		gguf_add_tensor(gguf, norm_weight);

		auto dst_file = dst / ("llama_layer_" + std::to_string(layer_idx) + ".gguf");
		gguf_write_to_file(gguf, dst_file.string().c_str(), false);
		gguf_free(gguf);

		ggml_free(layer_ctx);
	}

	std::cout << "Writing to file: " << dst << std::endl;
}

std::filesystem::path GetWeightFileName(int layer_idx, const std::string& weight_name)
{
	std::cout << "Loading weight: layer" << layer_idx << "." << weight_name << std::endl;
	std::filesystem::path model_folder = R"(D:\Python\Janus\model-file)";
	std::filesystem::path file_name = model_folder /
		(GetLayerWeightName(layer_idx, weight_name) + ".bin");
	return file_name;
}
