#include "language_model.h"

#include "quant.h"
#include "timer.h"

LlamaDecoderLayer::LlamaDecoderLayer(int layer_index, ggml_backend* container)
	: backend(container), layer_idx(layer_index)
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
		cached_k = std::make_shared<std::vector<uint8_t>>();
		cached_v = std::make_shared<std::vector<uint8_t>>();
		LoadFromFile("model-file/layer_" + std::to_string(layer_idx) + ".bin");
		return;
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

void LlamaDecoderLayer::FillTo(LlamaDecoderLayer& layer, bool async)
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
	layer.cached_k = cached_k;
	layer.cached_v = cached_v;
}

void LlamaDecoderLayer::SaveToFile(std::filesystem::path path)
{
	std::ofstream ofs(path, std::ios::binary);
	if (!ofs)
		throw std::runtime_error("Failed to open file for writing");
	ofs.write(reinterpret_cast<const char*>(input_norm_weight->data),
		ggml_nbytes(input_norm_weight));
	ofs.write(reinterpret_cast<const char*>(q_proj->data),
		ggml_nbytes(q_proj));
	ofs.write(reinterpret_cast<const char*>(k_proj->data),
		ggml_nbytes(k_proj));
	ofs.write(reinterpret_cast<const char*>(v_proj->data),
		ggml_nbytes(v_proj));
	ofs.write(reinterpret_cast<const char*>(o_proj->data),
		ggml_nbytes(o_proj));
	ofs.write(reinterpret_cast<const char*>(gate_proj->data),
		ggml_nbytes(gate_proj));
	ofs.write(reinterpret_cast<const char*>(up_proj->data),
		ggml_nbytes(up_proj));
	ofs.write(reinterpret_cast<const char*>(down_proj->data),
		ggml_nbytes(down_proj));
	ofs.write(reinterpret_cast<const char*>(norm_weight->data),
		ggml_nbytes(norm_weight));
}

void LlamaDecoderLayer::LoadFromFile(std::filesystem::path path)
{
	std::ifstream ifs(path, std::ios::binary);
	if (!ifs)
		throw std::runtime_error("Failed to open file for reading");
	ifs.read(reinterpret_cast<char*>(input_norm_weight->data),
		ggml_nbytes(input_norm_weight));
	ifs.read(reinterpret_cast<char*>(q_proj->data),
		ggml_nbytes(q_proj));
	ifs.read(reinterpret_cast<char*>(k_proj->data),
		ggml_nbytes(k_proj));
	ifs.read(reinterpret_cast<char*>(v_proj->data),
		ggml_nbytes(v_proj));
	ifs.read(reinterpret_cast<char*>(o_proj->data),
		ggml_nbytes(o_proj));
	ifs.read(reinterpret_cast<char*>(gate_proj->data),
		ggml_nbytes(gate_proj));
	ifs.read(reinterpret_cast<char*>(up_proj->data),
		ggml_nbytes(up_proj));
	ifs.read(reinterpret_cast<char*>(down_proj->data),
		ggml_nbytes(down_proj));
	ifs.read(reinterpret_cast<char*>(norm_weight->data),
		ggml_nbytes(norm_weight));
}

ggml_cgraph* LlamaDecoderLayer::build_llama_layer(size_t batch_size, size_t input_len)
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

		q = ggml_rope_inplace(ctx, q, pos_ids, head_dim, GGML_ROPE_TYPE_NEOX);
		k = ggml_rope_inplace(ctx, k, pos_ids, head_dim, GGML_ROPE_TYPE_NEOX);
		// ggml_permute 与 torch.permute 不一致
		// ggml_permute 接受的参数为源tensor维度的对应位置
		// torch.permute 接受的参数为目标tensor维度对应的位置
		// ggml_permute: dst->ne[permute] = src->ne
		// torch.permute: dst->ne = src->ne[permute]
		q = view_tensor(ctx, ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3)),
			head_dim, input_len, num_heads, batch_size);
		k = view_tensor(ctx, ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3)),
			head_dim, input_len, num_heads, batch_size);
		v = view_tensor(ctx, ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3)),
			input_len, head_dim, num_heads, batch_size);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_new");
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_new");

		// 连接 Cached K, V
		if (cached_length > 0)
		{
			auto past_k = ggml_new_tensor_4d(ctx, k->type,
				head_dim, cached_length, num_heads, batch_size);
			auto past_v = ggml_new_tensor_4d(ctx, v->type,
				cached_length, head_dim, num_heads, batch_size);
			ggml_set_name(past_k, "past_k");
			ggml_set_name(past_v, "past_v");

			k = ggml_concat(ctx, past_k, k, 1);
			v = ggml_concat(ctx, past_v, v, 0);
		}

		// 用法提示：ggml_mul_mat(ctx, a, b) => b * a^T
		// 返回的张量形状为 [ne3, ne2, b->ne[1], a->ne[1]]
		// 特此记录，以免忘记
		// Shape: [batch, num_head * head_dim, seq_len, seq_len]
		ggml_tensor* QK = ggml_mul_mat(ctx, k, q); // QK = Q * K^T
		// QK_scaled = QK / sqrt(dim)
		ggml_tensor* QK_scaled = ggml_scale_inplace(ctx, QK,
			1.0f / float(sqrt(head_dim)));
		// QK_masked = mask_past(QK_scaled)
		ggml_tensor* QK_masked = ggml_diag_mask_inf_inplace(
			ctx, QK_scaled, cached_length);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, QK_masked, "QK_masked");
		// QK = soft_max(QK_masked)
		ggml_tensor* QK_soft_max = ggml_soft_max_inplace(ctx, QK_masked);
		// attn_output = QK * V^T | Shape: [batch, num_head, seq_len, head_dim]
		auto attn_output = ggml_mul_mat(ctx, v, QK_soft_max);
		attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
		// Shape: [batch, seq_len, num_head * head_dim]
		attn_output = view_tensor(ctx, ggml_cont(ctx, attn_output),
			head_dim * num_heads, input_len, batch_size);
		attn_output = ggml_mul_mat(ctx, o_proj, attn_output);
		auto residual = ggml_add_inplace(ctx,
			flatten_tensor(ctx, attn_output), flatten_tensor(ctx, input_emb));
		residual = view_tensor(ctx, residual, 4096u, input_len, batch_size);

		// 层归一化
		auto mlp_input = ggml_rms_norm(ctx, residual, eps);
		mlp_input = ggml_mul_inplace(ctx, mlp_input, norm_weight);

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
	}
	ggml_build_forward_expand(layer_graph, output);
	ggml_free(ctx);
	return layer_graph;
}

std::vector<uint8_t> LlamaDecoderLayer::run_layer(
	std::vector<uint8_t>& input_embs_data,
	ggml_gallocr* layer_galloc,
	size_t batch_size, size_t input_len,
	bool save_details
) {
	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Layer);


	if (save_details) MidTensors::GetInstance().StartRegisterMidTensors();
	ModelTimer::GetInstance().Start(ModelTimer::TimerType::BuildGraph);
	auto gr = build_llama_layer(batch_size, input_len);
	ggml_gallocr_reserve(layer_galloc, gr);
	if (!ggml_gallocr_alloc_graph(layer_galloc, gr))
		throw std::runtime_error("Cannot allocate graph in LlamaDecoderLayer");
	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::BuildGraph);
	if (save_details) MidTensors::GetInstance().StopRegisterMidTensors();


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::CopyTensor);

	auto input_embs_tensor = ggml_graph_get_tensor(gr, "input_emb");
	ggml_backend_tensor_set(input_embs_tensor, input_embs_data.data(), 0, input_embs_data.size());
	auto pos_ids = ggml_graph_get_tensor(gr, "pos_ids");

	auto pos_ids_generator = std::views::iota(0, int32_t(input_len));
	std::vector<int> pos_ids_data(input_len);
	std::copy(pos_ids_generator.begin(), pos_ids_generator.end(), pos_ids_data.begin());
	ggml_backend_tensor_set(pos_ids, pos_ids_data.data(), 0, ggml_nbytes(pos_ids));

	if (cached_length > 0)
	{
		auto past_k = ggml_graph_get_tensor(gr, "past_k");
		auto past_v = ggml_graph_get_tensor(gr, "past_v");
		ggml_backend_tensor_set(past_k, cached_k->data(), 0, cached_k->size());
		ggml_backend_tensor_set(past_v, cached_v->data(), 0, cached_v->size());
	}

	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::CopyTensor);


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Compute);
	ggml_backend_graph_compute(backend, gr);
	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::Compute);


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::CopyTensor);

	auto output = ggml_graph_node(gr, -1);
	std::vector<uint8_t> result(ggml_nbytes(output));
	ggml_backend_tensor_get(output, result.data(), 0, result.size());

	auto k = ggml_graph_get_tensor(gr, "k_new");
	auto v = ggml_graph_get_tensor(gr, "v_new");
	auto offset = cached_k->size();
	auto append_size = ggml_nbytes(k);
	auto total_size = offset + append_size;
	cached_k->resize(total_size);
	cached_v->resize(total_size);
	ggml_backend_tensor_get(k, cached_k->data() + offset, 0, append_size);
	ggml_backend_tensor_get(v, cached_v->data() + offset, 0, append_size);

	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::CopyTensor);

	cached_length += int(input_len);

	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::Layer);


	if (save_details)
	{
		std::string backend_name;
		if (ggml_backend_is_cpu(backend))
			backend_name = "cpu";
		else if (ggml_backend_is_cuda(backend))
			backend_name = "cuda";
		else
			backend_name = "unknown";

		ggml_graph_dump_dot(gr, nullptr, (
			"inspect/" + backend_name + "/layer_" + std::to_string(layer_idx) + ".dot"
			).c_str());

		auto mid_tensors = MidTensors::GetInstance().get_mid_tensors(gr);
		for (auto tensor : mid_tensors)
		{
			auto file_name = "inspect/" + backend_name + "/" +
				std::string(tensor->name) + ".bin";
			std::ofstream ofs(file_name, std::ios::binary);
			while (!ofs.good())
			{
				std::cerr << "无法打开文件：" << file_name << std::endl;
				std::cout << "回车以重试" << std::endl;
				std::cin.get();
				ofs.open(file_name, std::ios::binary);
			}
			std::vector<char> data(ggml_nbytes(tensor));
			ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
			ofs.write(data.data(), data.size());
		}
	}

	return result;
}

LanguageModel::LanguageModel(
	bool load_layers, size_t gpu_offload_layer_num, int num_cpu_threads
)
	: cuda_backend(ggml_backend_cuda_init(0)),
	  cpu_backend(ggml_backend_cpu_init()),
	  gen_head(cpu_backend),
	  gpu_offload_num(gpu_offload_layer_num),
	  num_cpu_threads(num_cpu_threads)
{
	ggml_backend_cpu_set_n_threads(cpu_backend, num_cpu_threads);
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

std::vector<uint8_t> LanguageModel::preprocess(
	std::vector<int> input_ids, size_t parallel_size, size_t image_token_num_per_image
) {
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
	ggml_graph_compute_with_ctx(ctx, pre_graph, num_cpu_threads);

	std::vector<uint8_t> result(ggml_nbytes(inputs_embeds));
	memcpy(result.data(), inputs_embeds->data, result.size());
	ggml_free(ctx);
	return result;
}

std::vector<uint8_t> LanguageModel::postprocess(
	std::vector<uint8_t> hidden_states_data, size_t parallel_size, size_t input_len
) {
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
	ggml_graph_compute_with_ctx(ctx, post_graph, num_cpu_threads);
	std::vector<uint8_t> result(ggml_nbytes(out_states));
	memcpy(result.data(), out_states->data, result.size());
	ggml_free(ctx);
	return result;
}

std::vector<uint8_t> LanguageModel::run_model(
	std::vector<uint8_t> input_embs_data, size_t parallel_size, size_t input_len
) {
	auto cuda_ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda_backend));
	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Model);
	for (auto i : std::views::iota(0ull, gpu_offload_num))
	{
		// 运行模型
		auto& layer = offloads[i];
		input_embs_data = layer.run_layer(
			input_embs_data, cuda_ga, parallel_size * 2, input_len);
	}
	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::Model);
	ModelTimer::GetInstance().PrintTimeConsumedAll();
	auto mem_size = ggml_gallocr_get_buffer_size(cuda_ga, 0);
	std::cout << "CUDA Graph Memory Usage: " << mem_size << " bytes" << std::endl;
	ggml_gallocr_free(cuda_ga);

	input_embs_data = postprocess(input_embs_data, parallel_size, input_len);

	return input_embs_data;
}

std::pair<
	std::vector<uint8_t>, std::vector<uint8_t>
> LanguageModel::GenHead(
	std::vector<uint8_t> outputs, size_t parallel_size, size_t input_len)
{
	if (outputs.size() > 4096ull * parallel_size * 2) {
		// 取最后一个 token 的输出
		outputs.erase(outputs.begin(),
			outputs.end() - 4096ull * parallel_size * 2 * sizeof(float));
	}
	auto logits = gen_head.run_head(outputs, parallel_size);
	std::vector<uint8_t> logits_cond(logits.size() / 2);
	std::vector<uint8_t> logits_uncond(logits.size() / 2);
#pragma omp parallel for
	for (int i = 0; i < parallel_size; i++)
	{
		std::copy(logits.begin() + i * 16384ull * 2,
			logits.begin() + (i * 2 + 1) * 16384ull,
			logits_cond.begin() + i * 16384ull);
		std::copy(logits.begin() + (i * 2 + 1) * 16384ull,
			logits.begin() + (i + 1) * 16384ull * 2,
			logits_uncond.begin() + i * 16384ull);
	}
	return { logits_cond, logits_uncond };
}

std::vector<int> LanguageModel::sample_once(
	std::vector<uint8_t> cond, std::vector<uint8_t> uncond,
	size_t parallel_size, float temperature, float cfg_weight
) {
	auto cpu_ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cpu_backend));
	auto ctx = ggml_init({
		.mem_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE
				  + ggml_graph_overhead(),
		.no_alloc = true
		});

	auto gr = ggml_new_graph(ctx);
	auto cond_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16384ull, parallel_size);
	auto uncond_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 16384ull, parallel_size);
	cond_tensor = ggml_scale_inplace(ctx, cond_tensor, cfg_weight);
	uncond_tensor = ggml_scale_inplace(ctx, uncond_tensor, 1.0f - cfg_weight);
	auto logits = ggml_add_inplace(ctx, cond_tensor, uncond_tensor);
	auto probs_tensor = ggml_soft_max_inplace(ctx, logits);
	ggml_build_forward_expand(gr, probs_tensor);

	ggml_gallocr_reserve(cpu_ga, gr);
	ggml_gallocr_alloc_graph(cpu_ga, gr);
	ggml_backend_tensor_set(cond_tensor, cond.data(), 0, cond.size());
	ggml_backend_tensor_set(uncond_tensor, uncond.data(), 0, uncond.size());
	ggml_backend_graph_compute(cpu_backend, gr);

	std::vector<uint8_t> probs_data(ggml_nbytes(probs_tensor));
	ggml_backend_tensor_get(probs_tensor, probs_data.data(), 0, probs_data.size());
	ggml_free(ctx);
	ggml_gallocr_free(cpu_ga);

	// multinomial 采样
	auto probs = std::span(
		reinterpret_cast<float*>(probs_data.data()), probs_data.size() / 4);
	std::vector<int> sample_result(parallel_size);
	std::mt19937 gen(std::random_device{}());
	std::uniform_real_distribution<float> dist(0, 1);
	for (size_t i = 0; i < parallel_size; i++)
	{
		auto p = probs.subspan(i * 16384ull, 16384ull);
		std::sort(p.begin(), p.end(), std::greater<float>());
		auto r = dist(gen);
		int j = 0;
		while (r > p[j])
		{
			r -= p[j];
			j++;
		}
		sample_result[i] = j;
	}
	return sample_result;
}

std::vector<uint8_t> LanguageModel::gen_head_align(
	std::vector<int> tokens, size_t parallel_size
) {
	std::vector<int> double_tokens(tokens.size() * 2);
	for (size_t i = 0; i < tokens.size(); i++)
	{
		double_tokens[i * 2] = tokens[i];
		double_tokens[i * 2 + 1] = tokens[i];
	}
	return gen_head.embedding_mlp(tokens, parallel_size);
}

LanguageModel::GenHead::GenHead(ggml_backend* container)
{
	gen_head_backend = container;
	ggml_init_params gen_head_param = {
		.mem_size = ggml_tensor_overhead() * 5,
		.mem_buffer = nullptr,
		.no_alloc = true
	};
	gen_head_ctx = ggml_init(gen_head_param);
	output_mlp_projector = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 4096u, 4096u);
	vision_head = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 4096u, 16384u);
	mlp_p1 = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 8ull, 4096ull);
	mlp_p2 = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 4096ull, 4096ull);
	align_embeddings = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 8ull, 16384ull);
	ggml_backend_alloc_ctx_tensors(gen_head_ctx, container);
	auto buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\output_mlp_projector.bin)");
	ggml_backend_tensor_set(output_mlp_projector, buffer.data(), 0, buffer.size());
	buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\vision_head.bin)");
	ggml_backend_tensor_set(vision_head, buffer.data(), 0, buffer.size());
	buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\mlp_p1.bin)");
	ggml_backend_tensor_set(mlp_p1, buffer.data(), 0, buffer.size());
	buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\mlp_p2.bin)");
	ggml_backend_tensor_set(mlp_p2, buffer.data(), 0, buffer.size());
	buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\align_embeddings.bin)");
	ggml_backend_tensor_set(align_embeddings, buffer.data(), 0, buffer.size());
}

std::vector<uint8_t> LanguageModel::GenHead::run_head(
	std::vector<uint8_t> hidden_states_data, size_t parallel_size)
{
	auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gen_head_backend));
	ggml_init_params gen_head_param = {
		.mem_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE
				  + ggml_graph_overhead(),
		.no_alloc = true
	};
	auto ctx = ggml_init(gen_head_param);

	auto gr = ggml_new_graph(ctx);
	auto x0 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
		4096ull, 1, parallel_size * 2);
	auto x = ggml_mul_mat(ctx, output_mlp_projector, x0);
	x = ggml_gelu_inplace(ctx, x);
	x = ggml_mul_mat(ctx, vision_head, x);
	ggml_build_forward_expand(gr, x);

	ggml_gallocr_reserve(ga, gr);
	auto mem_size = ggml_gallocr_get_buffer_size(ga, 0);
	std::cout << "Gen Head memory size: " << std::fixed << std::setprecision(2)
		<< mem_size / 1024. / 1024 << std::endl;
	ggml_gallocr_alloc_graph(ga, gr);
	ggml_backend_tensor_set(x0, hidden_states_data.data(), 0, hidden_states_data.size());
	ggml_backend_graph_compute(gen_head_backend, gr);

	std::vector<uint8_t> result(ggml_nbytes(x));
	ggml_backend_tensor_get(x, result.data(), 0, result.size());
	ggml_free(ctx);
	ggml_gallocr_free(ga);
	return result;
}

std::vector<uint8_t> LanguageModel::GenHead::embedding_mlp(
	std::vector<int> tokens, size_t parallel_size
) {
	auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gen_head_backend));
	ggml_init_params gen_head_param = {
		.mem_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE
				  + ggml_graph_overhead(),
		.no_alloc = true
	};
	auto ctx = ggml_init(gen_head_param);

	auto gr = ggml_new_graph(ctx);
	auto ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, parallel_size * 2);
	auto embs = ggml_get_rows(ctx, align_embeddings, ids);
	embs = ggml_mul_mat(ctx, mlp_p1, embs);
	embs = ggml_gelu_inplace(ctx, embs);
	embs = ggml_mul_mat(ctx, mlp_p2, embs);
	ggml_build_forward_expand(gr, embs);

	ggml_gallocr_reserve(ga, gr);
	auto mem_size = ggml_gallocr_get_buffer_size(ga, 0);
	std::cout << "Gen Head memory size: " << std::fixed << std::setprecision(2)
		<< mem_size / 1024. / 1024 << std::endl;
	ggml_gallocr_alloc_graph(ga, gr);
	ggml_backend_tensor_set(ids, tokens.data(), 0, ggml_nbytes(ids));
	ggml_backend_graph_compute(gen_head_backend, gr);

	std::vector<uint8_t> result(ggml_nbytes(embs));
	ggml_backend_tensor_get(embs, result.data(), 0, result.size());
	ggml_free(ctx);
	ggml_gallocr_free(ga);
	return result;
}
