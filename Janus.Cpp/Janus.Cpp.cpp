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
#include <opencv2/opencv.hpp>

#ifndef _DEBUG
constexpr int num_threads = 16;
#else
constexpr int num_threads = 1;
#endif

#include "vq_model.h"
#include "language_model.h"
#include "timer.h"

std::vector<cv::Mat> decode_images(
	std::vector<int> token_ids, size_t num_imgs, size_t img_sz)
{
	// auto backend = ggml_backend_cuda_init(0);
	auto backend = ggml_backend_cpu_init();
	ggml_backend_cpu_set_n_threads(backend, 16);
	GenDecoder decoder{ backend };
	const size_t num_patchs_w = img_sz / 16;
	auto img = decoder.decode_img_tokens(token_ids, num_imgs, num_patchs_w);
	ggml_backend_free(backend);

	std::vector<cv::Mat> imgs;
	cv::Mat img_mat(int(img_sz), int(img_sz), CV_8UC3);
	auto float_span = std::span(reinterpret_cast<float*>(img.data()), img_sz * img_sz);
	for (auto& f : float_span)
		f = float(std::clamp((f + 1.) / 2 * 255, 0., 255.));
	const size_t channel_offset = img_sz * img_sz;
	if (img_sz > std::numeric_limits<int>::max())
		throw std::runtime_error("Img too BIG !!!");
#pragma omp parallel for
	for (int i = 0; i < int(img_sz); i++)
	{
		for (int j = 0; j < int(img_sz); j++)
		{
			auto idx = i * img_sz + j;
			img_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(
				(uint8_t)float_span[channel_offset * 2 + idx],
				(uint8_t)float_span[channel_offset * 1 + idx],
				(uint8_t)float_span[channel_offset * 0 + idx]);
		}
	}
	imgs.push_back(img_mat);
	return imgs;
}

std::vector<cv::Mat> generate(
	std::vector<uint8_t> embeddings,
	std::shared_ptr<LanguageModel> model,
	float temperature = 1,
	int num_imgs = 16,
	int img_size = 384,
	float cfg_weight = 5
)
{
	const int num_patchs = img_size * img_size / 256;
	const size_t input_len = embeddings.size() / 4096 / num_imgs / 2 / 4;

	std::vector<int> generated_tokens; // Shape: [576, parallel_size]
	std::vector<int> input_ids;
	// Pre-fill
	{
		std::cout << "Token     0";
		auto result = model->run_model(embeddings, num_imgs, input_len);
		auto [logits_cond, logits_uncond] =
			model->GenHead(result, num_imgs, input_len);
		input_ids = model->sample_once(
			logits_cond, logits_uncond, num_imgs, temperature, cfg_weight);
		generated_tokens.append_range(input_ids);
		std::cout << std::setw(8) << input_ids[0] << std::endl;
	}
	ModelTimer::GetInstance().ClearAll();
	for (auto token_num : std::views::iota(1, num_patchs))
	{
		std::cout << "Token " << std::setw(5) << token_num;
		auto embeddings = model->gen_head_align(input_ids, num_imgs);
		auto result = model->run_model(embeddings, num_imgs, 1);
		auto [logits_cond, logits_uncond] = model->GenHead(result, num_imgs, 1);
		input_ids = model->sample_once(
			logits_cond, logits_uncond, num_imgs, temperature, cfg_weight);
		generated_tokens.append_range(input_ids);
		std::cout << std::setw(8) << input_ids[0] << std::endl;
		ModelTimer::GetInstance().ClearAll();
	}

	return decode_images(generated_tokens, num_imgs, img_size);
}

void test()
{
	auto language_model = std::make_shared<LanguageModel>(true, 30, num_threads);
	std::vector<int> input{
		100000, 5726, 25, 207, 1615,
		29834, 66515, 8781, 18640, 612,
		8143, 29445, 62913, 398, 185,
		185, 77398, 25, 100016
	};
	auto embeddings = language_model->preprocess(input, 1, 576);
	// reinterpret_cast<float*>(embeddings.data())[0] = std::nanf("");
	auto result = language_model->run_model(embeddings, 1, 19);
	dump_data_retry(result, "inspect/model.bin");
}

int main(int argc, char** argv)
{
	{
		constexpr size_t num_imgs = 1;
		constexpr size_t img_sz = 384;
		constexpr size_t num_patchs = img_sz * img_sz / 256;
		auto language_model = std::make_shared<LanguageModel>(true, 30, num_threads);
		std::vector<int> input{
			100000, 5726, 25, 207, 1615,
			29834, 66515, 8781, 18640, 612,
			8143, 29445, 62913, 398, 185,
			185, 77398, 25, 100016
		};
		input[0] = 100000;
		auto embeddings = language_model->preprocess(input, num_imgs, num_patchs);
		auto img = generate(embeddings, language_model, 1, num_imgs, img_sz, 5);
		cv::imwrite("inspect/out.png", img[0]);
		return 0;
	}
	// test(); return -1;
	{
		auto model = new LanguageModel(true, 30);
		std::vector<int> input{
			100000, 5726, 25, 207, 1615,
			29834, 66515, 8781, 18640, 612,
			8143, 29445, 62913, 398, 185,
			185, 77398, 25, 100016
		};
		auto embeddings = model->preprocess(input, 1, 576);
		dump_data_retry(embeddings, "inspect/inputs_embeddings.bin");

		auto result = model->run_model(embeddings, 1, 19);
		dump_data_retry(result, "inspect/model.bin");
		//auto bcuda = ggml_backend_cuda_init(0);
		//auto bcpu = ggml_backend_cpu_init();
		//ggml_backend_cpu_set_n_threads(bcpu, num_threads);
		//LlamaDecoderLayer layer{ 0, nullptr };
		//LlamaDecoderLayer gpu_layer{ -1, bcuda };
		//LlamaDecoderLayer cpu_layer{ -1, bcpu };
		//layer.FillTo(gpu_layer);
		//layer.FillTo(cpu_layer);
		//auto acuda = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bcuda));
		//auto acpu = ggml_gallocr_new(ggml_backend_get_default_buffer_type(bcpu));
		//std::vector<uint8_t> result;

		//layer.ClearCache();
		//result = gpu_layer.run_layer(embeddings, acuda, 2, 19, true);
		//dump_data_retry(result, "inspect/layer-0-cuda.bin");

		//layer.ClearCache();
		//embeddings.erase(embeddings.begin(), embeddings.begin() + embeddings.size() / 2);
		//MidTensors::GetInstance().SetPathPrefix("cor-");
		//result = cpu_layer.run_layer(embeddings, acpu, 2, 19, true);
		//dump_data_retry(result, "inspect/layer-0-cpu.bin");

		auto [logits_cond, logits_uncond] =
			model->GenHead(result, 1, 19);
		dump_data_retry(logits_cond, "inspect/logits_cond.bin");
		dump_data_retry(logits_uncond, "inspect/logits_uncond.bin");
		auto input_ids = model->sample_once(
			logits_cond, logits_uncond, 1, 1, 5);
		for (auto id : input_ids)
			std::cout << id << ' ';
		std::endl(std::cout);
		return 0;
	}
	{
		ggml_context* ctx = ggml_init({
			.mem_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
			ggml_graph_overhead(),
			.no_alloc = true
			});
		auto gr = ggml_new_graph(ctx);
		auto cuda = ggml_backend_cuda_init(0);
		auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda));
		auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3, 2, 3);
		auto A = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
		ggml_backend_alloc_ctx_tensors(ctx, cuda);
		auto y = ggml_mul_mat(ctx, A, x);
		ggml_build_forward_expand(gr, y);
		ggml_gallocr_reserve(ga, gr);
		ggml_gallocr_alloc_graph(ga, gr);
		auto x_gen = std::views::iota(0, 18);
		std::vector<float> x_data(x_gen.begin(), x_gen.end());
		ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * 4);
		auto A_data = std::vector<float>{ 1, 0, 0, 0, 1, 0 };
		ggml_backend_tensor_set(A, A_data.data(), 0, A_data.size() * 4);
		ggml_backend_graph_compute(cuda, gr);
		std::vector<float> result(12);
		ggml_backend_tensor_get(y, result.data(), 0, result.size() * 4);
		for (auto i : result)
			std::cout << i << ' ';
		std::endl(std::cout);
		return -1;
	}
	{
		int input_length = 2;
		int element_num = 4096 * input_length;
		std::vector<uint8_t> embeddings;
		std::span<float> float_span;

		auto backend = ggml_backend_cuda_init(0);
		LlamaDecoderLayer layer{ 0, nullptr };
		LlamaDecoderLayer gpu_layer{ -1, backend };
		layer.FillTo(gpu_layer);
		auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
		std::vector<uint8_t> result;

		embeddings.resize(element_num * 4);
		float_span = std::span(reinterpret_cast<float*>(embeddings.data()), element_num);
		for (auto i : std::views::iota(0, element_num))
			float_span[i] = float(i) / element_num - 0.2f;
		gpu_layer.ClearCache();
		MidTensors::GetInstance().SetPathPrefix("stepped-1-");
		result = gpu_layer.run_layer(embeddings, ga, 1, input_length, true);
		char* critical = new char[32*4];
		std::copy(result.end() - 4096ull * 4,
			result.end() - 4096ull * 4 + 8 * 4ull, critical);
		std::copy(result.end() - 4096ull * 4 + 1024,
			result.end() - 4096ull * 4 + 8 * 4ull + 1024, critical + 8 * 4);
		for (auto i : std::views::iota(0, element_num))
			float_span[i] = float(i) / element_num - 0.7f;
		MidTensors::GetInstance().SetPathPrefix("stepped-2-");
		result = gpu_layer.run_layer(embeddings, ga, 1, input_length, true);
		std::copy(result.end() - 4096ull * 4,
			result.end() - 4096ull * 4 + 8 * 4ull, critical + 16 * 4);
		std::copy(result.end() - 4096ull * 4 + 1024,
			result.end() - 4096ull * 4 + 8 * 4ull + 1024, critical + 24 * 4);

		embeddings.resize(element_num * 2 * 4);
		float_span = std::span(reinterpret_cast<float*>(embeddings.data()), element_num * 2);
		for (auto i : std::views::iota(0, element_num * 2))
			if (i < element_num)
				float_span[i] = float(i) / element_num - 0.2f;
			else
				float_span[i] = float(i - element_num) / element_num - 0.7f;
		gpu_layer.ClearCache();
		MidTensors::GetInstance().SetPathPrefix("full-");
		result = gpu_layer.run_layer(embeddings, ga, 1, input_length * 2, true);
		char* more = new char[16 * 4];
		std::copy(result.end() - 4096ull * 4,
			result.end() - 4096ull * 4 + 8 * 4ull, more);
		std::copy(result.end() - 4096ull * 4 + 1024,
			result.end() - 4096ull * 4 + 8 * 4ull + 1024, more + 8 * 4);

		ggml_gallocr_free(ga);
		ggml_backend_free(backend);
		return 0;
	}
	{
		int input_length = 4;
		int element_num = 4096 * input_length;
		std::vector<uint8_t> embeddings(element_num * 4);
		auto float_span = std::span(reinterpret_cast<float*>(embeddings.data()), element_num);
		for (auto i : std::views::iota(0, element_num))
			float_span[i] = float(i) / element_num - 0.2f;
		std::vector<uint8_t> double_embeddings(embeddings.size() * 2);
		std::copy(embeddings.begin(), embeddings.end(), double_embeddings.begin());
		std::copy(embeddings.begin(), embeddings.end(),
			double_embeddings.begin() + embeddings.size());
		auto language_model = std::make_shared<LanguageModel>(true, 30, num_threads);
		// 2 x 4 x 4096 float
		auto result = language_model->run_model(double_embeddings, 1, input_length);
		return 0;
	}

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
