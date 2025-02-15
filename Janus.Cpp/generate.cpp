
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-cuda.h>

#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "ggml-base.lib")
#pragma comment(lib, "ggml-cpu.lib")
#pragma comment(lib, "ggml-blas.lib")
#pragma comment(lib, "ggml-cuda.lib")

#include <span>
#include <numeric>
#include <algorithm>

#include "language_model.h"
#include "vq_model.h"
#include "generate.h"

#include "timer.h"

std::vector<cv::Mat> decode_images(
	std::vector<std::vector<int>> batch_token_ids,
	size_t num_imgs, size_t img_sz)
{
	// auto backend = ggml_backend_cuda_init(0);
	auto backend = ggml_backend_cpu_init();
	ggml_backend_cpu_set_n_threads(backend, 16);
	GenDecoder decoder{ backend };
	const size_t num_patchs_w = img_sz / 16;
	const size_t num_tokens_per_img = num_patchs_w * num_patchs_w;

	std::vector<cv::Mat> imgs;
	for (auto& token_ids : batch_token_ids)
	{
		auto img = decoder.decode_img_tokens(token_ids, num_imgs, num_patchs_w);
		cv::Mat img_mat(int(img_sz), int(img_sz), CV_8UC3);
		auto float_span = std::span(
			reinterpret_cast<float*>(img.data()),
			img_sz * img_sz * 3);
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
				auto r = (uint8_t)float_span[channel_offset * 0 + idx];
				auto g = (uint8_t)float_span[channel_offset * 1 + idx];
				auto b = (uint8_t)float_span[channel_offset * 2 + idx];
				img_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
			}
		}
		imgs.push_back(img_mat);
	}

	ggml_backend_free(backend);
	return imgs;
}

std::vector<cv::Mat> generate(
	std::vector<uint8_t> embeddings,
	std::shared_ptr<LanguageModel> model,
	float temperature, int num_imgs,
	int img_size, float cfg_weight,
	std::optional<
		std::reference_wrapper<std::vector<std::vector<int>>>
	> output_tokens
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

	std::vector<std::vector<int>> batch_tokens(num_imgs);
	for (size_t i = 0; i < num_patchs; i++) {
		for (size_t j = 0; j < num_imgs; j++)
			batch_tokens[j].push_back(generated_tokens[i + j * img_size]);
	}
	output_tokens->get() = batch_tokens;

	return decode_images(batch_tokens, num_imgs, img_size);
}
