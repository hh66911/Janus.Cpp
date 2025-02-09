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
#include <mdspan>
#include <random>

#ifndef _DEBUG
constexpr int num_threads = 16;
#else
constexpr int num_threads = 1;
#endif

#include "language_model.h"
#include "timer.h"

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

	LlamaDecoderLayer layer_data{ 0 };
	auto cuda = ggml_backend_cuda_init(0);
	auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda));
	LlamaDecoderLayer gpu_layer{ -1, cuda };
	layer_data.FillTo(gpu_layer);
	gpu_layer.run_layer(emb, ga, 32, 16, false);
	emb.resize(emb.size() / 2);
	auto gpu_result = gpu_layer.run_layer(emb, ga, 32, 8, true);
	ggml_gallocr_free(ga);
	ggml_backend_free(cuda);
}

int main(int argc, char** argv)
{
	{
		auto language_model = new LanguageModel{ true, 30, num_threads };
		std::vector<int> input(16);
		for (auto i : std::views::iota(1, 16))
			input[i] = i;
		input[0] = 100000;
		auto embeddings = language_model->preprocess(input);

		std::vector<int> generated_tokens; // Shape: [576, parallel_size]
		// Pre-fill
		{
			auto hidden_states = language_model->run_model(embeddings, 16, 16);
			auto result = language_model->postprocess(hidden_states, 16, 16);
			auto [logits_cond, logits_uncond] = language_model->GenHead(result, 16, 16);
			input = language_model->sample_once(
				logits_cond, logits_uncond, 16, 1.f, 5.f);
			generated_tokens.append_range(input);
		}
		ModelTimer::GetInstance().ClearAll();
		for (auto token_num : std::views::iota(1, 576))
		{
			std::cout << "Token " << token_num << std::endl;
			auto embeddings = language_model->gen_head_align(input, 16);
			auto hidden_states = language_model->run_model(embeddings, 16, 1);
			auto result = language_model->postprocess(hidden_states, 16, 1);
			auto [logits_cond, logits_uncond] = language_model->GenHead(result, 16, 1);
			input = language_model->sample_once(
				logits_cond, logits_uncond, 16, 1.f, 5.f);
			generated_tokens.append_range(input);
			ModelTimer::GetInstance().ClearAll();
		}
		delete language_model;

		return 0;
	}
	{
		auto language_model = new LanguageModel(true, 30);
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
