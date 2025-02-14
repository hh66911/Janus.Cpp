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
				(uint8_t)float_span[channel_offset * 0 + idx],
				(uint8_t)float_span[channel_offset * 1 + idx],
				(uint8_t)float_span[channel_offset * 2 + idx]);
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

ggml_backend* cuda;
ggml_gallocr* ga;
LlamaDecoderLayer layer{ 0, nullptr };
LlamaDecoderLayer* gpulayer;
LanguageModel model{ false, 0, num_threads };

#include "temp.h"
int test1()
{
	std::vector<int> input{
		100000, 5726, 25, 207, 1615,
		29834, 66515, 8781, 18640, 612,
		8143, 29445, 62913, 398, 185,
		185, 77398, 25, 100016
	};
	MidTensors::GetInstance().SetPathPrefix("emb21/");
	auto emb0 = model.preprocess(input, 1, 576);
	MidTensors::GetInstance().dump_data_retry(emb0, "inspect/inputs/emb0.bin");
	auto emb1 = model.gen_head_align({ data_tokens[0] }, 1);
	MidTensors::GetInstance().dump_data_retry(emb1, "inspect/inputs/emb1.bin");
	emb1 = contact_embeddings(emb0, emb1, 19, 1, 2);
	MidTensors::GetInstance().dump_data_retry(emb1, "inspect/inputs/emb1_cat.bin");
	auto emb2 = model.gen_head_align({ data_tokens[1] }, 1);
	MidTensors::GetInstance().dump_data_retry(emb2, "inspect/inputs/emb2.bin");
	emb2 = contact_embeddings(emb1, emb2, 20, 1, 2);
	MidTensors::GetInstance().dump_data_retry(emb2, "inspect/inputs/emb2_cat.bin");
	auto result = gpulayer->run_layer(emb2, ga, 2, 21, true);
	return -1;
}

int test0()
{
	std::vector<int> input{
		100000, 5726, 25, 207, 1615,
		29834, 66515, 8781, 18640, 612,
		8143, 29445, 62913, 398, 185,
		185, 77398, 25, 100016
	};
	auto emb0 = model.preprocess(input, 1, 576);
	std::vector<uint8_t> result;
	MidTensors::GetInstance().SetPathPrefix("emb0/");
	result = gpulayer->run_layer(emb0, ga, 2, 19, true);
	// dump_data_retry(result, "inspect/gen/emb0_out.bin");

	int idx = 1;
	for (auto i : data_tokens)
	{
		if (idx == 3) break;
		MidTensors::GetInstance().SetPathPrefix("emb" + std::to_string(idx) + "/");
		auto emb = model.gen_head_align({ i }, 1);
		result = gpulayer->run_layer(emb, ga, 2, 1, true);
		// dump_data_retry(result, "inspect/gen/emb" + std::to_string(idx) + "_out.bin");
		idx++;
	}
	return -1;
}

int main(int argc, char** argv)
{
	//cuda = ggml_backend_cuda_init(0);
	//ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cuda));
	//gpulayer = new LlamaDecoderLayer{ -1, cuda };
	//layer.FillTo(*gpulayer);
	//test0();
	//gpulayer->ClearCache();
	//return test1();
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
	auto embeddings = language_model->preprocess(input, num_imgs, num_patchs);
	auto img = generate(embeddings, language_model, 1, num_imgs, img_sz, 5);
	cv::imwrite("inspect/out.png", img[0]);
	return 0;
}
