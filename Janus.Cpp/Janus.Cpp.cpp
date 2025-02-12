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
	ggml_backend_cpu_set_n_threads(backend, num_threads);
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
	int image_token_num_per_image = 576,
	int img_size = 384,
	float cfg_weight = 5
)
{
	const int num_patchs = img_size * img_size / 256;
	const size_t input_len = embeddings.size() / 4096 / num_imgs / 2;

	std::vector<int> generated_tokens; // Shape: [576, parallel_size]
	std::vector<int> input_ids;
	// Pre-fill
	{
		auto result = model->run_model(embeddings, num_imgs, input_len);
		auto [logits_cond, logits_uncond] =
			model->GenHead(result, num_imgs, input_len);
		input_ids = model->sample_once(
			logits_cond, logits_uncond, num_imgs, temperature, cfg_weight);
		generated_tokens.append_range(input_ids);
	}
	ModelTimer::GetInstance().ClearAll();
	for (auto token_num : std::views::iota(1, num_patchs))
	{
		std::cout << "Token " << token_num << std::endl;
		auto embeddings = model->gen_head_align(input_ids, num_imgs);
		auto result = model->run_model(embeddings, num_imgs, 1);
		auto [logits_cond, logits_uncond] = model->GenHead(result, num_imgs, 1);
		input_ids = model->sample_once(
			logits_cond, logits_uncond, num_imgs, temperature, cfg_weight);
		generated_tokens.append_range(input_ids);
		ModelTimer::GetInstance().ClearAll();
	}

	return decode_images(generated_tokens, num_imgs, img_size);
}

int main(int argc, char** argv)
{
	{
		std::vector<uint8_t> embeddings(4096 * 4);
		auto float_span = std::span(reinterpret_cast<float*>(embeddings.data()), 4096);
		for (auto i : std::views::iota(0, 4096))
			float_span[i] = i / 4096.f - 0.2f;
		std::vector<uint8_t> double_embeddings(embeddings.size() * 2);
		std::copy(embeddings.begin(), embeddings.end(), double_embeddings.begin());
		std::copy(embeddings.begin(), embeddings.end(),
			double_embeddings.begin() + embeddings.size());
		auto language_model = std::make_shared<LanguageModel>(true, 30, num_threads);
		auto result = language_model->run_model(double_embeddings, 1, 1);
		return 0;
	}
	{
		std::vector<uint8_t> embeddings(4096 * 4);
		auto float_span = std::span(reinterpret_cast<float*>(embeddings.data()), 4096);
		for (auto i : std::views::iota(0, 4096))
			float_span[i] = i / 4096.f - 0.2f;
		auto backend = ggml_backend_cuda_init(0);
		LlamaDecoderLayer layer{ 0, nullptr };
		LlamaDecoderLayer gpu_layer{ -1, backend };
		layer.FillTo(gpu_layer);
		auto ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
		auto result = gpu_layer.run_layer(embeddings, ga, 1, 1, true);
		ggml_gallocr_free(ga);
		ggml_backend_free(backend);
		return 0;
	}
	{
		constexpr size_t num_imgs = 1;
		constexpr size_t img_sz = 128;
		constexpr size_t num_patchs = img_sz * img_sz / 256;
		auto language_model = std::make_shared<LanguageModel>(true, 30, num_threads);
		std::vector<int> input{
			100000, 5726, 25, 37727, 11, 946,
			418, 340, 30, 185, 185, 77398, 25,
			100016
		};
		input[0] = 100000;
		auto embeddings = language_model->preprocess(input, num_imgs, num_patchs);
		auto img = generate(embeddings, language_model, 1, num_imgs, num_patchs, img_sz, 5);
		cv::imwrite("inspect/out.png", img[0]);
		return 0;
	}
	{
		auto language_model = new LanguageModel(true, 30);
		std::vector<int> input(16);
		for (auto i : std::views::iota(1, 16))
			input[i] = i;
		input[0] = 100000;
		auto embeddings = language_model->preprocess(input, 16, 576);
		auto result = language_model->run_model(embeddings, 16, 16);
		delete language_model;
		std::cout << "Writing result to file\n";
		std::ofstream file("inspect/result.bin", std::ios::binary);
		if (!file.is_open())
			throw std::runtime_error("Failed to open file: result.bin");
		file.write(reinterpret_cast<const char*>(result.data()), result.size());
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
