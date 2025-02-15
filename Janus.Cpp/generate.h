#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#ifndef _DEBUG
constexpr int num_threads = 16;
#else
constexpr int num_threads = 1;
#endif

std::vector<cv::Mat> decode_images(
	std::vector<int> batch_token_ids, size_t num_imgs, size_t img_sz);

std::vector<cv::Mat> generate(
	std::vector<uint8_t> embeddings,
	std::shared_ptr<LanguageModel> model,
	float temperature = 1,
	int num_imgs = 16,
	int img_size = 384,
	float cfg_weight = 5
);
