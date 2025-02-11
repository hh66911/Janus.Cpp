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
	ggml_build_forward_expand(gr, target);
	mid_tensors.push_back({
		target, ctx, gr
	});
}

void MidTensors::SaveMidTensors(const std::string& path)
{
	for (auto tensor : GetTensors())
	{
		auto file_name = path / std::filesystem::path(std::string(tensor->name) + ".bin");
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
	}
}
