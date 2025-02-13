#include "tensor_utils.h"

#include <filesystem>
#include <string>
#include <fstream>

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
