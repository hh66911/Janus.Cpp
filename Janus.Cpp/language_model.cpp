﻿#include "language_model.h"

#include "quant.h"
#include "timer.h"

RemoteLayer::RemoteLayer(std::string endpoint)
	: client(endpoint)
{}

void RemoteLayer::LoadRange(std::pair<size_t, size_t> start_end)
{
	httplib::Params params;
	params.emplace("start", std::to_string(start_end.first));
	params.emplace("end", std::to_string(start_end.second));
	auto res = client.Get("/load_range", params, {});
	if (res && res->status == 200)
		std::cout << "Load range: " << start_end.first << " - " << start_end.second << std::endl;
	else
		throw std::runtime_error("Failed to load range");
}

std::vector<uint8_t> RemoteLayer::Run(
	const std::vector<uint8_t>& embeddings, size_t batch_size, size_t input_len)
{
	auto res = client.Get("/set_params", httplib::Params{
		{"batch_size", std::to_string(batch_size)},
		{"input_len", std::to_string(input_len)}
		}, {});
	if (res && res->status != 200)
		throw std::runtime_error("Failed to set params");

	res = client.Post(
		"/run", embeddings.size(),
		[&embeddings, batch_size, input_len](size_t offset, size_t length, httplib::DataSink& sink) {
			const char* data = reinterpret_cast<const char*>(embeddings.data());
			sink.write(data + offset, length);
			return true; // return 'false' if you want to cancel the request.
		}, "application/octet-stream");
	if (res && res->status == 200)
	{
		std::vector<uint8_t> result(res->body.size());
		std::copy(res->body.begin(), res->body.end(), result.begin());
		return result;
	}
	else
		throw std::runtime_error("Failed to run layer");
}

std::vector<uint8_t> RemoteLayer::RefillBatch(
	const std::vector<uint8_t>& embeddings, size_t batch_idx)
{
	auto res = client.Get("/set_params", httplib::Params{
		{"batch_idx", std::to_string(batch_idx)}
		}, {});
	if (res && res->status != 200)
		throw std::runtime_error("Failed to set batch index");

	res = client.Post(
		"/refill_batch", embeddings.size(),
		[&embeddings, batch_idx](size_t offset, size_t length, httplib::DataSink& sink) {
			const char* data = reinterpret_cast<const char*>(embeddings.data());
			sink.write(data + offset, length);
			return true; // return 'false' if you want to cancel the request.
		}, "application/octet-stream");
	if (res && res->status == 200)
	{
		std::vector<uint8_t> result(res->body.size());
		std::copy(res->body.begin(), res->body.end(), result.begin());
		return result;
	}
	else
		throw std::runtime_error("Failed to refill batch");
}

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
		// ~400 MB when type = BF16
		auto mem_size = num_elements * ggml_type_size(type) / ggml_blck_size(type)
			+ 4096 * ggml_type_size(GGML_TYPE_F32) * 2
			+ num_tensors * ggml_tensor_overhead();
		ctx_buffer.resize(mem_size);
		ggml_init_params layer_param = {
			.mem_size = mem_size,
			.mem_buffer = ctx_buffer.data(),
			.no_alloc = true
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

void LlamaDecoderLayer::QuantLayer(
	int layer_idx, ggml_backend* quant_end,
	std::filesystem::path src, std::filesystem::path dst)
{
	LlamaDecoderLayer layer{ -1, quant_end };

	auto base_name = "layers." + std::to_string(layer_idx) + ".";
	RawF32TensorFromFile(layer.layer_ctx, layer.input_norm_weight, src / (base_name + "input_layernorm.weight.bin"));
	RawF32TensorFromFile(layer.layer_ctx, layer.norm_weight, src / (base_name + "post_attention_layernorm.weight.bin"));
	QuantTensorFromFile(layer.layer_ctx, layer.q_proj, src / (base_name + "self_attn.q_proj.weight.bin"));
	QuantTensorFromFile(layer.layer_ctx, layer.k_proj, src / (base_name + "self_attn.k_proj.weight.bin"));
	QuantTensorFromFile(layer.layer_ctx, layer.v_proj, src / (base_name + "self_attn.v_proj.weight.bin"));
	QuantTensorFromFile(layer.layer_ctx, layer.o_proj, src / (base_name + "self_attn.o_proj.weight.bin"));
	QuantTensorFromFile(layer.layer_ctx, layer.gate_proj, src / (base_name + "mlp.gate_proj.weight.bin"));
	QuantTensorFromFile(layer.layer_ctx, layer.up_proj, src / (base_name + "mlp.up_proj.weight.bin"));
	QuantTensorFromFile(layer.layer_ctx, layer.down_proj, src / (base_name + "mlp.down_proj.weight.bin"));

	layer.SaveToFile(dst / (base_name + "quant.bin"));
}

LlamaDecoderLayer LlamaDecoderLayer::FromQuanted(
	int layer_idx, ggml_backend* backend, std::filesystem::path folder)
{
	LlamaDecoderLayer layer{ layer_idx, backend };
	if (layer_idx < 0)
		throw std::runtime_error("Invalid layer index");

	layer.cached_k = std::make_shared<std::vector<uint8_t>>();
	layer.cached_v = std::make_shared<std::vector<uint8_t>>();
	layer.LoadFromFile(folder / ("layers." + std::to_string(layer_idx) + ".quant.bin"));

	return layer;
}

ggml_cgraph* LlamaDecoderLayer::build_llama_layer(
	size_t batch_size, size_t input_len, bool use_cache, bool fast_attn)
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

	auto input_emb = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
		4096u, input_len, batch_size);
	ggml_set_name(input_emb, "input_emb");

	auto pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_len);
	ggml_set_name(pos_ids, "pos_ids");

	ggml_tensor* output = nullptr;
	{
		// 层归一化 Shape: [batch, seq_len, 4096]
		auto rms_normed_input = ggml_rms_norm(ctx, input_emb, eps);
		rms_normed_input = ggml_mul_inplace(ctx, rms_normed_input, input_norm_weight);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, rms_normed_input, "rms_normed_input");

		auto q = ggml_mul_mat(ctx, q_proj, rms_normed_input);
		auto k = ggml_mul_mat(ctx, k_proj, rms_normed_input);
		auto v = ggml_mul_mat(ctx, v_proj, rms_normed_input);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_raw");

		// ggml_permute 与 torch.permute 不一致
		// ggml_permute 接受的参数为源tensor维度的对应位置
		// torch.permute 接受的参数为目标tensor维度对应的位置
		// ggml_permute: dst->ne[permute] = src->ne
		// torch.permute: dst->ne = src->ne[permute]
		// 调整形状到 [batch, seq_len, num_head, head_dim]
		q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
		q = view_tensor(ctx, q, head_dim, num_heads * batch_size, input_len);
		k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
		k = view_tensor(ctx, k, head_dim, num_heads * batch_size, input_len);
		v = view_tensor(ctx, v, head_dim, num_heads, input_len, batch_size);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_new_raw");

		v = ggml_permute(ctx, v, 0, 1, 3, 2); // [seq_len, batch, num_head, head_dim]
		ggml_build_forward_expand(layer_graph, ggml_set_name(ggml_cont(ctx, v), "v_next_token"));
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_new_permuted");

		// 注意！！！当使用CUDA作为GGML的Backend时，NeoX的RoPE操作不会遍历batch维度！！！
		// 解决方案：将batch维度和num_head维度合并，RoPE之后再拆分
		q = ggml_rope_inplace(ctx, q, pos_ids, head_dim, GGML_ROPE_TYPE_NEOX);
		k = ggml_rope_inplace(ctx, k, pos_ids, head_dim, GGML_ROPE_TYPE_NEOX);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, q, "q_rope_raw");
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_rope_raw");
		q = view_tensor(ctx, q, head_dim, num_heads, batch_size, input_len);
		k = view_tensor(ctx, k, head_dim, num_heads, batch_size, input_len);
		ggml_build_forward_expand(layer_graph, ggml_set_name(ggml_cont(ctx, k), "k_next_token"));
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_new_permuted");
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, q, "q_new_permuted");

		// 连接 Cached K, V
		if (use_cache && cached_length > 0)
		{
			auto past_k = ggml_new_tensor_4d(ctx, k->type,
				head_dim, num_heads, batch_size, cached_length);
			auto past_v = ggml_new_tensor_4d(ctx, v->type,
				head_dim, num_heads, batch_size, cached_length);
			ggml_set_name(past_k, "past_k");
			ggml_set_name(past_v, "past_v");

			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, past_k, "past_k1");
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, past_v, "past_v1");

			k = ggml_concat(ctx, past_k, k, 3);
			v = ggml_concat(ctx, past_v, v, 3);

			if (!worst_case_enabled)
			{
				// 用于防止重复分配内存
				auto kv_buffer = ggml_new_tensor_4d(ctx, k->type,
					head_dim, num_heads, batch_size * 2, max_cached_length - cached_length);
				ggml_build_forward_expand(layer_graph, kv_buffer);
				auto attn_buffer = ggml_new_tensor_4d(ctx, k->type,
					max_cached_length - cached_length,
					max_cached_length - cached_length,
					num_heads, batch_size * 2);
				ggml_build_forward_expand(layer_graph, attn_buffer);
				worst_case_enabled = true;
			}
		}

		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_cat");
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_cat");

		ggml_tensor* attn_output = nullptr;
		if (fast_attn && cached_length > 0)
		{
			const size_t padded_cachelen = GGML_PAD(k->ne[3], 256u);
			k = view_tensor(ctx, ggml_pad(ctx,
				view_tensor(ctx, k, head_dim * num_heads * batch_size, k->ne[3]),
				0, int(padded_cachelen - k->ne[3]), 0, 0),
				head_dim, num_heads, batch_size, padded_cachelen);
			v = view_tensor(ctx, ggml_pad(ctx,
				view_tensor(ctx, v, head_dim * num_heads * batch_size, v->ne[3]),
				0, int(padded_cachelen - v->ne[3]), 0, 0),
				head_dim, num_heads, batch_size, padded_cachelen);
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_cat_pad");
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_cat_pad");

			k = ggml_cast(ctx, k, GGML_TYPE_F16);
			v = ggml_cast(ctx, v, GGML_TYPE_F16);

			const size_t padded_inplen = GGML_PAD(input_len, GGML_KQ_MASK_PAD);

			// [batch_size, num_heads, input_len, head_dim]
			v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 3, 1));
			k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 3, 1));
			q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 3, 1));

			auto mask_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, padded_cachelen, padded_inplen);
			ggml_set_input(ggml_set_name(mask_tensor, "mask_input"));
			std::vector<ggml_tensor*> attn_batched;
			for (auto i : std::views::iota(0ull, batch_size))
			{
				auto qi = ggml_view_4d(ctx, q, head_dim, input_len, num_heads, 1,
					q->nb[1], q->nb[2], q->nb[3], q->nb[3] * i);
				auto ki = ggml_view_4d(ctx, k, head_dim, padded_cachelen, num_heads, 1,
					k->nb[1], k->nb[2], k->nb[3], k->nb[3] * i);
				auto vi = ggml_view_4d(ctx, v, head_dim, padded_cachelen, num_heads, 1,
					v->nb[1], v->nb[2], v->nb[3], v->nb[3] * i);
				MidTensors::GetInstance().inspect_tensor(
					ctx, layer_graph, qi, "q[" + std::to_string(i) + "]");
				MidTensors::GetInstance().inspect_tensor(
					ctx, layer_graph, ki, "k[" + std::to_string(i) + "]");
				MidTensors::GetInstance().inspect_tensor(
					ctx, layer_graph, vi, "v[" + std::to_string(i) + "]");
				auto output_i = ggml_flash_attn_ext(ctx,
					qi, ki, vi, mask_tensor, 1.f / float(sqrt(head_dim)), 0, 0);
				MidTensors::GetInstance().inspect_tensor(
					ctx, layer_graph, output_i, "attn_output[" + std::to_string(i) + "]");
				attn_batched.push_back(output_i);
			}
			// 连接所有的 residual
			if (attn_batched.size() > 1)
			{
				typedef std::span<ggml_tensor*> tensors;
				auto merge_all = [ctx](tensors cont) -> ggml_tensor* {
					auto merge = [ctx](this auto& self, tensors l, tensors r) -> ggml_tensor* {
						if (l.size() == 1) return ggml_concat(ctx, l[0], r[0], 3);
						auto mid = l.size() / 2;
						auto lmerge = self(l.subspan(0, mid), r.subspan(0, mid));
						auto rmerge = self(l.subspan(mid), r.subspan(mid));
						return ggml_concat(ctx, lmerge, rmerge, 3);
					};
					assert(cont.size() % 2 == 0);
					auto mid = cont.size() / 2;
					return merge(cont.subspan(0, mid), cont.subspan(mid));
				};
				attn_output = merge_all(attn_batched);
			}
			else
				attn_output = attn_batched[0];
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, attn_output, "res_merged");
		}
		else
		{
			// 用法提示：ggml_mul_mat(ctx, a, b) => b * a^T
			// 返回的张量形状为 [ne3, ne2, b->ne[1], a->ne[1]]
			// 特此记录，以免忘记
			// Shape: [batch, num_head * head_dim, seq_len, seq_len]

			// [batch, num_head, seq_len, head_dim]
			q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 3, 1));
			k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 3, 1));

			// [batch, num_head, head_dim, seq_len]
			v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 3, 0));

			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_cat_perm");
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_cat_perm");
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, q, "q_cur");

			ggml_tensor* QK = ggml_mul_mat(ctx, k, q); // QK = Q * K^T
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, QK, "QK");
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
			attn_output = ggml_mul_mat(ctx, v, QK_soft_max);
		}

		attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
		// Shape: [batch, seq_len, num_head * head_dim]
		attn_output = view_tensor(ctx, ggml_cont(ctx, attn_output),
			head_dim * num_heads, input_len, batch_size);
		attn_output = ggml_mul_mat(ctx, o_proj, attn_output);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, attn_output, "o_proj");
		auto residual = ggml_add_inplace(ctx, flatten_tensor(ctx, attn_output), flatten_tensor(ctx, input_emb));
		residual = view_tensor(ctx, residual, head_dim * num_heads, input_len, batch_size);

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
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, output, "output");
	}
	ggml_build_forward_expand(layer_graph, output);
	ggml_free(ctx);
	return layer_graph;
}

std::vector<uint8_t> LlamaDecoderLayer::run_layer(
	const std::vector<uint8_t>& input_embs_data,
	ggml_gallocr* layer_galloc,
	size_t batch_size, size_t input_len,
	bool save_details, bool use_cache
) {
	if (input_embs_data.size() != 4096 * input_len * batch_size * 4)
		throw std::runtime_error("Input size mismatch");
	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Layer);


	if (save_details) MidTensors::GetInstance().StartRegisterMidTensors();
	ModelTimer::GetInstance().Start(ModelTimer::TimerType::BuildGraph);
	auto gr = build_llama_layer(batch_size, input_len, use_cache, use_flash_attn);
	ggml_gallocr_reserve(layer_galloc, gr);
	if (!ggml_gallocr_alloc_graph(layer_galloc, gr))
		throw std::runtime_error("Cannot allocate graph in LlamaDecoderLayer");
	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::BuildGraph);
	if (save_details) MidTensors::GetInstance().StopRegisterMidTensors();


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::CopyTensor);

	auto input_embs_tensor = ggml_graph_get_tensor(gr, "input_emb");
	ggml_backend_tensor_set(input_embs_tensor, input_embs_data.data(), 0, input_embs_data.size());

	auto pos_ids = ggml_graph_get_tensor(gr, "pos_ids");
	auto pos_ids_generator = std::views::iota(
		cached_length, cached_length + int(input_len));
	std::vector<int> pos_ids_data(input_len);
	std::copy(pos_ids_generator.begin(), pos_ids_generator.end(), pos_ids_data.begin());
	ggml_backend_tensor_set(pos_ids, pos_ids_data.data(), 0, ggml_nbytes(pos_ids));

	if (cached_length > 0)
	{
		if (use_flash_attn)
		{
			auto mask_tensor = ggml_graph_get_tensor(gr, "mask_input");
			size_t padded_k = mask_tensor->ne[0];
			size_t padded_len = mask_tensor->ne[1];
			std::vector<ggml_fp16_t> mask_data(padded_len * padded_k);
			const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-1e10);
			const ggml_fp16_t zero16 = ggml_fp32_to_fp16(0);
			for (auto i : std::views::iota(0ull, input_len))
			{
				auto mask = mask_data.begin() + i * padded_k;
				std::fill(mask, mask + cached_length + i + 1, zero16);
				std::fill(mask + cached_length + i + 1, mask + padded_k, neg_inf);
			}
			std::fill(mask_data.begin() + padded_k * input_len, mask_data.end(), neg_inf);
			ggml_backend_tensor_set(mask_tensor, mask_data.data(), 0, ggml_nbytes(mask_tensor));
		}

		auto past_k = ggml_graph_get_tensor(gr, "past_k");
		auto past_v = ggml_graph_get_tensor(gr, "past_v");
		ggml_backend_tensor_set(past_k, cached_k->data(), 0, ggml_nbytes(past_k));
		ggml_backend_tensor_set(past_v, cached_v->data(), 0, ggml_nbytes(past_v));
	}

	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::CopyTensor);


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Compute);
	ggml_backend_graph_compute(backend, gr);
	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::Compute);


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::CopyTensor);

	auto output = ggml_graph_node(gr, -1);
	std::vector<uint8_t> result(ggml_nbytes(output));
	ggml_backend_tensor_get(output, result.data(), 0, result.size());

	auto k = ggml_graph_get_tensor(gr, "k_next_token");
	auto v = ggml_graph_get_tensor(gr, "v_next_token");
	const auto total_size = ggml_nbytes(k);
	const auto batch_token_bytes = head_dim * num_heads * 4ull * batch_size;
	auto size_cur = cached_k->size(), offset_pos = cached_length * batch_token_bytes;
	if (size_cur < total_size + offset_pos)
	{
		const auto incre = (input_len % cache_incre == 0 ?
			input_len : (input_len - (input_len % cache_incre) + cache_incre));
		cache_capacity += incre;
		cached_k->resize(size_cur + incre * batch_token_bytes);
		cached_v->resize(size_cur + incre * batch_token_bytes);
	}
	ggml_backend_tensor_get(k, cached_k->data() + offset_pos, 0, total_size);
	ggml_backend_tensor_get(v, cached_v->data() + offset_pos, 0, total_size);

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

		MidTensors::GetInstance().dump_data_retry(
			*cached_k, "inspect/" + backend_name + "/layer_" +
			std::to_string(layer_idx) + "/cached_k.bin");
		MidTensors::GetInstance().dump_data_retry(
			*cached_v, "inspect/" + backend_name + "/layer_" +
			std::to_string(layer_idx) + "/cached_v.bin");

		ggml_graph_dump_dot(gr, nullptr, (
			"inspect/" + backend_name + "/layer_" + std::to_string(layer_idx) + ".dot"
			).c_str());

		MidTensors::GetInstance().SaveMidTensors(
			"inspect/" + backend_name + "/layer_" + std::to_string(layer_idx) + "/");
	}

	return result;
}

ggml_cgraph* LlamaDecoderLayer::build_refill_graph(size_t input_len, bool fast_attn)
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

	auto input_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4096u, input_len);
	ggml_set_name(input_emb, "input_emb");

	auto pos_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_len);
	ggml_set_name(pos_ids, "pos_ids");

	ggml_tensor* output = nullptr;
	{
		// 层归一化 Shape: [seq_len, 4096]
		auto rms_normed_input = ggml_rms_norm(ctx, input_emb, eps);
		rms_normed_input = ggml_mul_inplace(ctx, rms_normed_input, input_norm_weight);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, rms_normed_input, "rms_normed_input");

		auto q = ggml_mul_mat(ctx, q_proj, rms_normed_input);
		auto k = ggml_mul_mat(ctx, k_proj, rms_normed_input);
		auto v = ggml_mul_mat(ctx, v_proj, rms_normed_input);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_raw");

		// 调整形状到 [seq_len, num_head, head_dim]
		q = view_tensor(ctx, q, head_dim, num_heads, input_len);
		k = view_tensor(ctx, k, head_dim, num_heads, input_len);
		v = view_tensor(ctx, v, head_dim, num_heads, input_len);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_new_raw");

		ggml_build_forward_expand(layer_graph, ggml_set_name(ggml_cont(ctx, v), "v_next_token"));

		q = ggml_rope_inplace(ctx, q, pos_ids, head_dim, GGML_ROPE_TYPE_NEOX);
		k = ggml_rope_inplace(ctx, k, pos_ids, head_dim, GGML_ROPE_TYPE_NEOX);
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, q, "q_rope_raw");
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_rope_raw");
		ggml_build_forward_expand(layer_graph, ggml_set_name(ggml_cont(ctx, k), "k_next_token"));

		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_cat");
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_cat");

		// [num_head, seq_len, head_dim]
		q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
		k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

		// [num_head, head_dim, seq_len]
		v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, k, "k_cat_perm");
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, v, "v_cat_perm");
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, q, "q_cur");

		ggml_tensor* residual = nullptr;
		if (fast_attn)
		{
			// ggml_flash_attn_ext()
		}
		else
		{
			// 用法提示：ggml_mul_mat(ctx, a, b) => b * a^T
			// 返回的张量形状为 [ne3, ne2, b->ne[1], a->ne[1]]
			// 特此记录，以免忘记
			// Shape: [batch, num_head * head_dim, seq_len, seq_len]
			ggml_tensor* QK = ggml_mul_mat(ctx, k, q); // QK = Q * K^T
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, QK, "QK");
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
			attn_output = view_tensor(ctx, ggml_cont(ctx, attn_output), head_dim * num_heads, input_len);
			attn_output = ggml_mul_mat(ctx, o_proj, attn_output);
			MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, attn_output, "o_proj");
			residual = ggml_add_inplace(ctx, flatten_tensor(ctx, attn_output), flatten_tensor(ctx, input_emb));
			residual = view_tensor(ctx, residual, 4096u, input_len);
		}

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
		MidTensors::GetInstance().inspect_tensor(ctx, layer_graph, output, "output");
	}
	ggml_build_forward_expand(layer_graph, output);
	ggml_free(ctx);
	return layer_graph;
}

std::vector<uint8_t> LlamaDecoderLayer::refill_batch(
	const std::vector<uint8_t>& input_embs_data,
	ggml_gallocr* layer_galloc, size_t batch_idx
) {
	if (input_embs_data.size() != 4096 * cached_length * 4)
		throw std::runtime_error("Input size mismatch");
	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Layer);


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::BuildGraph);
	auto gr = build_refill_graph(cached_length, false);
	ggml_gallocr_reserve(layer_galloc, gr);
	if (!ggml_gallocr_alloc_graph(layer_galloc, gr))
		throw std::runtime_error("Cannot allocate graph in LlamaDecoderLayer");
	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::BuildGraph);


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::CopyTensor);

	auto input_embs_tensor = ggml_graph_get_tensor(gr, "input_emb");
	ggml_backend_tensor_set(input_embs_tensor, input_embs_data.data(), 0, input_embs_data.size());
	auto pos_ids = ggml_graph_get_tensor(gr, "pos_ids");

	auto pos_ids_generator = std::views::iota(0, cached_length);
	std::vector<int> pos_ids_data(cached_length);
	std::copy(pos_ids_generator.begin(), pos_ids_generator.end(), pos_ids_data.begin());
	ggml_backend_tensor_set(pos_ids, pos_ids_data.data(), 0, ggml_nbytes(pos_ids));

	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::CopyTensor);


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Compute);
	ggml_backend_graph_compute(backend, gr);
	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::Compute);


	ModelTimer::GetInstance().Start(ModelTimer::TimerType::CopyTensor);

	auto output = ggml_graph_node(gr, -1);
	std::vector<uint8_t> result(ggml_nbytes(output));
	ggml_backend_tensor_get(output, result.data(), 0, result.size());

	auto k = ggml_graph_get_tensor(gr, "k_next_token");
	auto v = ggml_graph_get_tensor(gr, "v_next_token");
	const auto token_size = head_dim * num_heads * 4ull;
	const auto batched_token_size = cached_k->size() / cache_capacity;
	const auto batch_size = batched_token_size / token_size;
	std::vector<uint8_t> buffer(token_size * cached_length);
	ggml_backend_tensor_get(k, buffer.data(), 0, ggml_nbytes(k));
	for (auto t : std::views::iota(0, cached_length))
	{
		auto offset = batch_idx * token_size + t * batched_token_size;
		std::copy(buffer.begin() + t * token_size,
			buffer.begin() + (t + 1) * token_size,
			cached_k->data() + offset);
	}
	ggml_backend_tensor_get(v, buffer.data(), 0, ggml_nbytes(v));
	for (auto t : std::views::iota(0, cached_length))
	{
		auto offset = batch_idx * token_size + t * batched_token_size;
		std::copy(buffer.begin() + t * token_size,
			buffer.begin() + (t + 1) * token_size,
			cached_v->data() + offset);
	}

	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::CopyTensor);


	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::Layer);

	return result;
}

LayerServer::LayerServer(std::string endpoint, std::filesystem::path model_file)
	: server(), batch_idx(0), batch_size(0), input_len(0), model_file(model_file)
{
	auto host = endpoint.substr(0, endpoint.find(':'));
	auto port = endpoint.substr(endpoint.find(':') + 1);
	server.bind_to_port(host, std::stoi(port));
	server.Get("/load_range",
		[this](const httplib::Request& req, httplib::Response& res) {
			auto start = req.get_param_value("start");
			auto end = req.get_param_value("end");
			if (start.empty() || end.empty())
			{
				res.status = 400;
				return;
			}
			std::pair<size_t, size_t> range;
			auto iss = std::istringstream(start);
			iss >> range.first;
			iss = std::istringstream(end);
			iss >> range.second;
			LoadRange(range);
		});

	server.Get("/set_params",
		[this](const httplib::Request& req, httplib::Response& res) {
			auto batch_size = req.get_param_value("batch_size");
			auto input_len = req.get_param_value("input_len");
			auto batch_idx = req.get_param_value("batch_idx");

			if (batch_idx.empty())
			{
				if (batch_size.empty() || input_len.empty())
				{
					res.status = 400;
					return;
				}
				auto iss = std::istringstream(batch_size);
				iss >> this->batch_size;
				iss = std::istringstream(input_len);
				iss >> this->input_len;
				std::cout << "Set params: batch_size = " << batch_size
					<< ", input_len = " << input_len << std::endl;
			}
			else
			{
				if (batch_size.empty() && input_len.empty())
				{
					auto iss = std::istringstream(batch_idx);
					iss >> this->batch_idx;
					std::cout << "Set batch index: " << batch_idx << std::endl;
				}
				else
				{
					res.status = 400;
					return;
				}
			}
		});

	server.Post("/run",
		[this](
			const httplib::Request& req,
			httplib::Response& res,
			const httplib::ContentReader& reader
		) -> void {
			std::vector<uint8_t> embs;
			reader([&](const char* data, size_t data_length) {
				auto offset = embs.size();
				embs.resize(offset + data_length);
				std::copy(data, data + data_length, embs.data() + offset);
				return true;
			});
			if (embs.empty())
			{
				res.status = 400;
				return;
			}
			try
			{
				embs = Run(embs);
			}
			catch (std::runtime_error e)
			{
				std::cerr << e.what();
				res.status = 400;
				return;
			}
			res.set_content(
				reinterpret_cast<char*>(embs.data()),
				embs.size(), "application/octet-stream");
		});

	server.Post("/refill_batch",
		[this](
			const httplib::Request & req,
			httplib::Response & res,
			const httplib::ContentReader & reader
		) -> void {
			std::vector<uint8_t> embs;
			reader([&](const char* data, size_t data_length) {
				auto offset = embs.size();
				embs.resize(offset + data_length);
				std::copy(data, data + data_length, embs.data() + offset);
				return true;
			});
			if (embs.empty())
			{
				res.status = 400;
				return;
			}
			try
			{
				embs = RefillBatch(embs);
			}
			catch (std::runtime_error e)
			{
				std::cerr << e.what();
				res.status = 400;
				return;
			}
			res.set_content(
				reinterpret_cast<char*>(embs.data()),
				embs.size(), "application/octet-stream");
		});
}

void LayerServer::StartServer()
{
	server.listen_after_bind();
}

void LayerServer::StopServer()
{
}

void LayerServer::LoadRange(std::pair<size_t, size_t> range)
{
	layers.clear();
	if (backend)
	{
		ggml_gallocr_free(ga);
		ggml_backend_free(backend);
	}
	backend = ggml_backend_cuda_init(0);
	ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
	auto cpu_backend = ggml_backend_cpu_init();
	layers.reserve(range.second - range.first);
	auto folder = model_file / "quanted_layers";
	for (auto i : std::views::iota(range.first, range.second))
	{
		auto layer = LlamaDecoderLayer::FromQuanted((int)i, cpu_backend, folder);
		layers.emplace_back(-1, backend);
		layer.FillTo(layers.back());
	}
	ggml_backend_free(cpu_backend);
}

std::vector<uint8_t> LayerServer::Run(std::vector<uint8_t> emb)
{
	for (auto& layer : layers)
		emb = layer.run_layer(emb, ga, batch_size, input_len);
	return emb;
}

std::vector<uint8_t> LayerServer::RefillBatch(std::vector<uint8_t> embs)
{
	for (auto& layer : layers)
		embs = layer.refill_batch(embs, ga, batch_idx);
	return embs;
}

LanguageModel::LanguageModel(
	size_t remote_num, int num_cpu_threads
)
	: cuda_backend(ggml_backend_cuda_init(0)),
	  cpu_backend(ggml_backend_cpu_init()),
	  gen_head(cpu_backend),
	  remote_num(remote_num),
	  remote_range{ (30 - remote_num + 1) / 2, (30 + remote_num + 1) / 2 },
	  num_cpu_threads(num_cpu_threads)
{
	ggml_backend_cpu_set_n_threads(cpu_backend, num_cpu_threads);
	ggml_init_params model_params = {
		.mem_size = ggml_tensor_overhead() * 2 +
			4096ull * 102400 * ggml_type_size(GGML_TYPE_F32) +
			4096ull * ggml_type_size(GGML_TYPE_F32),
		.mem_buffer = nullptr,
		.no_alloc = false
	};
	model_ctx = ggml_init(model_params);

	cuda_ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda_backend));

	// 从文件加载
	input_embeddings = ggml_new_tensor_2d(
		model_ctx, GGML_TYPE_F32, 4096u, 102400u);
	output_rms_norm = ggml_new_tensor_1d(model_ctx, GGML_TYPE_F32, 4096ull);
}

LanguageModel LanguageModel::LoadFromBin(
	size_t remote_num,
	int num_cpu_threads,
	std::filesystem::path src_folder
)
{
	LanguageModel model{ remote_num, num_cpu_threads };
	RawF32TensorFromFile(model.model_ctx, model.input_embeddings,
		R"(D:\Python\Janus\model-file\embed_tokens.bin)");
	RawF32TensorFromFile(model.model_ctx, model.output_rms_norm,
		R"(D:\Python\Janus\model-file\norm.weight.bin)");

	model.layers.resize(30);
#pragma omp parallel for
	for (int i = 0; i < 30; i++)
	{
		if (i >= model.remote_range.first && i < model.remote_range.second)
			continue;
		model.layers[i] = std::make_unique<LlamaDecoderLayer>(
			LlamaDecoderLayer::FromQuanted(i, model.cpu_backend, src_folder / "quanted_layers")
		);
		if (model.layers[i]->layer_idx == -1)
			_ASSERT(false);
	}

	model.offloads.reserve(30 - remote_num);

	for (auto i : std::views::iota(0ull, model.remote_range.first))
	{
		model.offloads.emplace_back(-1, model.cuda_backend);
		model.layers[i]->FillTo(model.offloads[i]);
	}

	for (auto i : std::views::iota(model.remote_range.second, 30ull))
	{
		model.offloads.emplace_back(-1, model.cuda_backend);
		model.layers[i]->FillTo(model.offloads[i - remote_num]);
	}

	if (remote_num > 0)
	{
		model.remote_layer = RemoteLayer("49.68.229.162:9800");
		model.remote_layer->LoadRange(model.remote_range);
	}

	return model;
}

std::vector<uint8_t> LanguageModel::get_pad_embs(
	size_t input_len, bool with_sentence_start, bool with_img_start)
{
	std::vector<int> tokens_data(input_len);
	std::fill(tokens_data.begin(), tokens_data.end(), pad_id);
	if (with_sentence_start) tokens_data[0] = 100000;
	if (with_img_start) tokens_data.back() = 100016;

	ggml_init_params builder_params = {
		.mem_size = (tokens_data.size() * 4097) * 4 +
			ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
	};
	auto ctx = ggml_init(builder_params);
	auto pre_graph = ggml_new_graph(ctx);

	auto tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, input_len);
	std::copy(tokens_data.begin(), tokens_data.end(), static_cast<int*>(tokens->data));
	// 获取输入嵌入
	ggml_tensor* inputs_embeds = ggml_get_rows(ctx, input_embeddings, tokens);

	ggml_build_forward_expand(pre_graph, inputs_embeds);
	ggml_graph_compute_with_ctx(ctx, pre_graph, num_cpu_threads);

	std::vector<uint8_t> result(ggml_nbytes(inputs_embeds));
	memcpy(result.data(), inputs_embeds->data, result.size());
	ggml_free(ctx);
	return result;
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
			std::fill(tokens_data.begin() + i * input_len,
				tokens_data.begin() + (i + 1) * input_len ,
				pad_id);
			if (processed_length == 0)
				tokens_data[i * input_len] = input_ids[0];
			tokens_data[(i + 1) * input_len - 1] = input_ids.back();
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
	out_states = ggml_mul_inplace(ctx, out_states, output_rms_norm);
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
	std::vector<uint8_t> input_embs_data,
	size_t parallel_size,
	size_t input_len,
	bool dump_data
) {
	if (input_embs_data.size() != parallel_size * 2 * input_len * 4096ull * 4)
		throw std::runtime_error("Input embeddings size mismatch");
	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Model);

	for (auto i : std::views::iota(0ull, remote_range.first))
	{
		// 运行模型
		input_embs_data = offloads[i].run_layer(
			input_embs_data, cuda_ga, parallel_size * 2, input_len);
		if (dump_data)
			MidTensors::GetInstance().dump_data_retry(
				input_embs_data, "inspect/model/layer_" + std::to_string(i) + ".bin");
	}
	if (remote_layer)
		input_embs_data = remote_layer->Run(input_embs_data, parallel_size * 2, input_len);
	for (auto i : std::views::iota(remote_range.second, 30ull))
	{
		// 运行模型
		input_embs_data = offloads[i - remote_num].run_layer(
			input_embs_data, cuda_ga, parallel_size * 2, input_len);
		if (dump_data)
			MidTensors::GetInstance().dump_data_retry(
				input_embs_data, "inspect/model/layer_" + std::to_string(i) + ".bin");
	}

	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::Model);
	// ModelTimer::GetInstance().PrintTimeConsumedAll();
	auto mem_size = ggml_gallocr_get_buffer_size(cuda_ga, 0);
	// std::cout << "CUDA Graph Memory Usage: " << mem_size << " bytes" << std::endl;

	input_embs_data = postprocess(input_embs_data, parallel_size, input_len);
	if (dump_data)
		MidTensors::GetInstance().dump_data_retry(
			input_embs_data, "inspect/model/postprocess.bin");

	processed_length += input_len;

	return input_embs_data;
}

void LanguageModel::refill_batch(
	std::vector<uint8_t> input_embs_data, size_t batch_idx)
{
	ModelTimer::GetInstance().Start(ModelTimer::TimerType::Model);
	for (auto i : std::views::iota(0ull, remote_range.first))
	{
		// 运行模型
		auto& layer = offloads[i];
		input_embs_data = layer.refill_batch(input_embs_data, cuda_ga, batch_idx);
	}
	if (remote_layer)
		input_embs_data = remote_layer->RefillBatch(input_embs_data, batch_idx);
	for (auto i : std::views::iota(remote_range.second, 30ull))
	{
		// 运行模型
		auto& layer = offloads[i - remote_num];
		input_embs_data = layer.refill_batch(input_embs_data, cuda_ga, batch_idx);
	}
	ModelTimer::GetInstance().Stop(ModelTimer::TimerType::Model);
}

std::pair<
	std::vector<uint8_t>, std::vector<uint8_t>
> LanguageModel::run_gen_head(
	std::vector<uint8_t> outputs, size_t parallel_size, size_t input_len)
{
	if (outputs.size() != parallel_size * 2 * 4096ull * input_len * 4)
		throw std::runtime_error("Output size mismatch");

	if (input_len > 1) {
		constexpr size_t N_offset = 4096ull * 4;
		const size_t B_offset = input_len * N_offset;
		// 取最后一个 token 的输出
		outputs.erase(outputs.begin(), outputs.end() - B_offset - N_offset);
		outputs.erase(outputs.begin() + N_offset, outputs.end() - N_offset);
	}
	auto logits = gen_head.run_head(outputs, parallel_size);
	std::vector<uint8_t> logits_cond(logits.size() / 2);
	std::vector<uint8_t> logits_uncond(logits.size() / 2);
	auto plogits = reinterpret_cast<float*>(logits.data());
	auto pcond = reinterpret_cast<float*>(logits_cond.data());
	auto puncond = reinterpret_cast<float*>(logits_uncond.data());
	for (int i = 0; i < parallel_size; i++)
	{
		memcpy(pcond, plogits, 16384 * sizeof(float));
		memcpy(puncond, plogits + 16384, 16384 * sizeof(float));
		plogits += 32768; pcond += 16384; puncond += 16384;
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
	probs_tensor = ggml_scale_inplace(ctx, probs_tensor, 1.0f / temperature);
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
#pragma omp parallel for
	for (int i = 0; i < int(parallel_size); i++)
	{
		auto p = probs.subspan(i * 16384ull, 16384ull);
		auto pos_gen = std::views::iota(0, 16384);
		std::vector<int> pos(pos_gen.begin(), pos_gen.end());
		std::sort(pos.begin(), pos.end(), [&](int a, int b) {
			return p[a] > p[b];
			});
		auto r = dist(gen);
		size_t j = 0;
		while (r > p[pos[j]])
			r -= p[pos[j++]];
		sample_result[i] = pos[j];
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
	return gen_head.embedding_mlp(double_tokens, parallel_size);
}

LanguageModel::GenHead::GenHead(ggml_backend* container)
{
	gen_head_backend = container;
	ggml_init_params gen_head_param = {
		.mem_size = ggml_tensor_overhead() * 7,
		.mem_buffer = nullptr,
		.no_alloc = true
	};
	gen_head_ctx = ggml_init(gen_head_param);
	ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(gen_head_backend));

	output_mlp_projector = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 4096u, 4096u);
	vision_head = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 4096u, 16384u);
	mlp_p1 = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 8ull, 4096ull);
	mlp_p2 = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F16, 4096ull, 4096ull);
	mlp_p1_bias = ggml_new_tensor_1d(gen_head_ctx, GGML_TYPE_F32, 4096ull);
	mlp_p2_bias = ggml_new_tensor_1d(gen_head_ctx, GGML_TYPE_F32, 4096ull);
	align_embeddings = ggml_new_tensor_2d(gen_head_ctx, GGML_TYPE_F32, 8ull, 16384ull);
	ggml_backend_alloc_ctx_tensors(gen_head_ctx, container);
	auto buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\output_mlp_projector.bin)");
	ggml_backend_tensor_set(output_mlp_projector, buffer.data(), 0, buffer.size());
	buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\vision_head.bin)");
	ggml_backend_tensor_set(vision_head, buffer.data(), 0, buffer.size());
	buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\mlp_p1.bin)");
	ggml_backend_tensor_set(mlp_p1, buffer.data(), 0, buffer.size());
	buffer = F16DataFromFile(R"(D:\Python\Janus\model-file\mlp_p2.bin)");
	ggml_backend_tensor_set(mlp_p2, buffer.data(), 0, buffer.size());
	buffer = F32DataFromFile(R"(D:\Python\Janus\model-file\mlp_p1_bias.bin)");
	ggml_backend_tensor_set(mlp_p1_bias, buffer.data(), 0, buffer.size());
	buffer = F32DataFromFile(R"(D:\Python\Janus\model-file\mlp_p2_bias.bin)");
	ggml_backend_tensor_set(mlp_p2_bias, buffer.data(), 0, buffer.size());
	buffer = RawF32DataFromFile(R"(D:\Python\Janus\model-file\align_embeddings.bin)");
	ggml_backend_tensor_set(align_embeddings, buffer.data(), 0, buffer.size());
}

std::vector<uint8_t> LanguageModel::GenHead::run_head(
	std::vector<uint8_t> hidden_states_data, size_t parallel_size)
{
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
	// auto mem_size = ggml_gallocr_get_buffer_size(ga, 0);
	// std::cout << "Gen Head memory size: " << std::fixed << std::setprecision(2)
	// 	<< mem_size / 1024. / 1024 << std::endl;
	ggml_gallocr_alloc_graph(ga, gr);
	ggml_backend_tensor_set(x0, hidden_states_data.data(), 0, hidden_states_data.size());
	ggml_backend_graph_compute(gen_head_backend, gr);

	std::vector<uint8_t> result(ggml_nbytes(x));
	ggml_backend_tensor_get(x, result.data(), 0, result.size());
	ggml_free(ctx);
	return result;
}

std::vector<uint8_t> LanguageModel::GenHead::embedding_mlp(
	std::vector<int> tokens, size_t parallel_size
) {
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
	embs = ggml_add_inplace(ctx, embs, mlp_p1_bias);
	embs = ggml_gelu_inplace(ctx, embs);
	embs = ggml_mul_mat(ctx, mlp_p2, embs);
	embs = ggml_add_inplace(ctx, embs, mlp_p2_bias);
	ggml_build_forward_expand(gr, embs);

	ggml_gallocr_reserve(ga, gr);
	// auto mem_size = ggml_gallocr_get_buffer_size(ga, 0);
	// std::cout << "Gen Head memory size: " << std::fixed << std::setprecision(2)
	// 	<< mem_size / 1024. / 1024 << std::endl;
	ggml_gallocr_alloc_graph(ga, gr);
	ggml_backend_tensor_set(ids, tokens.data(), 0, ggml_nbytes(ids));
	ggml_backend_graph_compute(gen_head_backend, gr);

	std::vector<uint8_t> result(ggml_nbytes(embs));
	ggml_backend_tensor_get(embs, result.data(), 0, result.size());
	ggml_free(ctx);
	return result;
}
