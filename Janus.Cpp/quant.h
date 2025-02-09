#pragma once

#include <fstream>
#include <filesystem>
#include <ranges>

#include <ggml.h>
#include "gguf.h"

void QuantTensorFromFile(ggml_context* ctx,
	ggml_tensor* dst, std::filesystem::path path);
void F32TensorFromFile(ggml_context* ctx,
	ggml_tensor* dst, std::filesystem::path path);
void F16TensorFromFile(ggml_context* ctx,
	ggml_tensor* dst, std::filesystem::path path);
std::vector<uint8_t> F16DataFromFile(std::filesystem::path path);
std::filesystem::path GetWeightFileName(int layer_idx, const std::string& weight_name);

inline std::string GetLayerWeightName(int layer_idx, const std::string& weight_name) {
	return "layer" + std::to_string(layer_idx) + "." + weight_name + ".weight";
}

void ConvertModelFile(std::filesystem::path src, std::filesystem::path dst);