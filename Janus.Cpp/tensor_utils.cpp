#include "tensor_utils.h"

#include <filesystem>
#include <string>
#include <fstream>
#include <mdspan>

std::vector<ggml_tensor*> MidTensors::get_mid_tensors(ggml_cgraph* gr)
{
	auto mid_tensor_get = mid_tensors
		| std::views::filter([gr](const auto& info) {
			return info.graph == gr;
		})
		| std::views::transform([](const auto& info) {
			return info.tensor;
		});
	std::vector<ggml_tensor*> mid_tensors(
		mid_tensor_get.begin(), mid_tensor_get.end()
	);
	return mid_tensors;
}

void MidTensors::inspect_tensor(
	ggml_context* ctx, ggml_cgraph* gr,
	ggml_tensor* target, const char* name
) {
	if (!enable_register)
		return;
	target = ggml_cont(ctx, target);
	ggml_set_name(target, name);
	if (gr == nullptr) gr = mid_tensors.back().graph;
	ggml_build_forward_expand(gr, target);
	mid_tensors.push_back({
		target, ctx, gr
		});
}

void MidTensors::inspect_tensor(
	ggml_context* ctx, ggml_cgraph* gr,
	ggml_tensor* target, const std::string& name
) {
	if (!enable_register)
		return;
	target = ggml_cont(ctx, target);
	ggml_set_name(target, name.c_str());
	if (gr == nullptr) gr = mid_tensors.back().graph;
	ggml_build_forward_expand(gr, target);
	mid_tensors.push_back({
		target, ctx, gr
		});
}

void MidTensors::SaveMidTensors(const std::string& path)
{
	std::stringstream oss;
	for (auto tensor : GetTensors())
	{
		auto file_name = path / std::filesystem::path(
			path_prefix + std::string(tensor->name) + ".bin");
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

		oss << std::setw(20) << tensor->name << "    ";
		oss << "Shape: [" << tensor->ne[3] << ", " << tensor->ne[2] << ", ";
		oss << tensor->ne[1] << ", " << tensor->ne[0] << "]" << std::endl;
	}
	std::ofstream shapes(path / std::filesystem::path(
		path_prefix + "shapes.txt"));
	shapes << oss.str();
}

std::vector<uint8_t> contact_embeddings(
	const std::vector<uint8_t>& left, const std::vector<uint8_t>& right,
	size_t left_len, size_t right_len, size_t batch_size)
{
	auto size_dst = left.size() + right.size();
	if (size_dst != (left_len + right_len) * 4096 * batch_size * 4)
		throw std::runtime_error("Invalid size");
	std::vector<uint8_t> result(size_dst);
	auto left_view = std::mdspan(
		reinterpret_cast<const float*>(left.data()),
		batch_size, left_len, 4096ull);
	auto right_view = std::mdspan(
		reinterpret_cast<const float*>(right.data()),
		batch_size, right_len, 4096ull);
	auto result_view = std::mdspan(
		reinterpret_cast<float*>(result.data()),
		batch_size, left_len + right_len, 4096ull);
	for (size_t i = 0; i < batch_size; i++)
	{
		for (size_t j = 0; j < left_len; j++)
			for (size_t k = 0; k < 4096; k++)
				result_view[std::array{ i, j, k }] = left_view[std::array{ i, j, k }];
		for (size_t j = 0; j < right_len; j++)
			for (size_t k = 0; k < 4096; k++)
				result_view[std::array{ i, j + left_len, k }] = right_view[std::array{ i, j, k }];
	}
	return result;
}
