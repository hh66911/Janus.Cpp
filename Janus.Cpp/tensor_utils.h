#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <iostream>
#include <vector>
#include <ranges>
#include <fstream>

std::vector<uint8_t> contact_embeddings(
	const std::vector<uint8_t>& left, const std::vector<uint8_t>& right,
	size_t left_len, size_t right_len, size_t batch_size);

inline void dump_data_retry(std::vector<uint8_t>& data, std::string name)
{
	std::ofstream ofs(name, std::ios::binary);
	while (!ofs.good())
	{
		std::cerr << "Failed to open file: " << name << std::endl;
		std::cout << "Press Enter to retry" << std::endl;
		std::cin.get();
		ofs.open(name, std::ios::binary);
	}
	ofs.write(reinterpret_cast<char*>(data.data()), data.size());
}

inline ggml_tensor* swish(ggml_context* ctx, ggml_tensor* x) {
	auto x_sig = ggml_sigmoid(ctx, x);
	return ggml_mul(ctx, x, x_sig);
}

inline ggml_tensor* swish_inplace(ggml_context* ctx, ggml_tensor* x) {
	auto x_sig = ggml_sigmoid(ctx, x);
	return ggml_mul_inplace(ctx, x, x_sig);
}

void inline print_shape(const ggml_tensor* tensor)
{
	std::cout << "Shape: [";
	for (auto i : std::views::iota(0, GGML_MAX_DIMS))
		std::cout << tensor->ne[i] << (i == GGML_MAX_DIMS - 1 ? "" : ", ");
	std::cout << "]\n";
}

inline void print_tensor_2d(const ggml_tensor* tensor)
{
	if (tensor->type == GGML_TYPE_I32)
	{
		for (auto i : std::views::iota(0, tensor->ne[1]))
		{
			for (auto j : std::views::iota(0, tensor->ne[0]))
				std::cout << static_cast<int*>(tensor->data)[i * tensor->ne[1] + j] << " ";
			std::cout << std::endl;
		}
	}
	else if (tensor->type == GGML_TYPE_F32)
	{
		for (auto i : std::views::iota(0, tensor->ne[1]))
		{
			for (auto j : std::views::iota(0, tensor->ne[0]))
				std::cout << static_cast<float*>(tensor->data)[i * tensor->ne[1] + j] << " ";
			std::cout << std::endl;
		}
	}
	else
	{
		std::cout << "Unsupported type\n";
	}
}

template <typename... DimTypes>
	requires (std::is_integral_v<DimTypes> && ...)
inline ggml_tensor* view_tensor(ggml_context* ctx, ggml_tensor* tensor, DimTypes... dims)
{
	if (!ggml_is_contiguous(tensor))
		throw std::runtime_error("Tensor view source must be contiguous");
	constexpr size_t num_dims = sizeof...(DimTypes);
	static_assert(num_dims <= GGML_MAX_DIMS);
	auto type = tensor->type;
	std::array<size_t, num_dims> ne = { dims... };
	if constexpr (num_dims == 1)
		return ggml_view_1d(ctx, tensor, ne[0], 0);
	else if constexpr (num_dims == 2)
		return ggml_view_2d(ctx, tensor, ne[0], ne[1],
			ggml_row_size(type, ne[0]), 0);
	else if constexpr (num_dims == 3)
		return ggml_view_3d(ctx, tensor, ne[0], ne[1], ne[2],
			ggml_row_size(type, ne[0]),
			ggml_row_size(type, ne[0] * ne[1]),
			0);
	else if constexpr (num_dims == 4)
		return ggml_view_4d(ctx, tensor, ne[0], ne[1], ne[2], ne[3],
			ggml_row_size(type, ne[0]),
			ggml_row_size(type, ne[0] * ne[1]),
			ggml_row_size(type, ne[0] * ne[1] * ne[2]),
			0);
	else
		static_assert(false);
}

inline ggml_tensor* flatten_tensor(ggml_context* ctx, ggml_tensor* tensor)
{
	auto ne = ggml_nelements(tensor);
	return ggml_view_1d(ctx, tensor, ne, 0);
}

class MidTensors
{
public:
	struct TensorInfo
	{
		ggml_tensor* tensor;
		ggml_context* ctx;
		ggml_cgraph* graph;
	};
	std::string path_prefix = "";

private:
	inline auto GetTensors() {
		return mid_tensors | std::views::transform([](const auto& info) {
			return info.tensor;
			});
	}
	inline auto GetContexts() {
		return mid_tensors | std::views::transform([](const auto& info) {
			return info.ctx;
			});
	}
	inline auto GetGraphs() {
		return mid_tensors | std::views::transform([](const auto& info) {
			return info.graph;
			});
	}

private:
	std::vector<TensorInfo> mid_tensors;
	bool enable_register = false;

private:
	MidTensors() = default;
	MidTensors(const MidTensors&) = delete;
	MidTensors& operator=(const MidTensors&) = delete;

public:
	static MidTensors& GetInstance() {
		static MidTensors instance;
		return instance;
	}

public:
	std::vector<ggml_tensor*> get_mid_tensors(ggml_cgraph* gr);

	void inspect_tensor(
		ggml_context* ctx, ggml_cgraph* gr,
		ggml_tensor* target, const char* name
	);
	void inspect_tensor(
		ggml_context* ctx, ggml_cgraph* gr,
		ggml_tensor* target, const std::string& name
	);

	inline void StartRegisterMidTensors() {
		enable_register = true;
		ClearMidTensors(false);
	}

	inline void StopRegisterMidTensors() {
		enable_register = false;
	}

	inline void ClearMidTensors(bool reset_prefix = true) {
		mid_tensors.clear();
		if (reset_prefix) path_prefix.clear();
	}

	inline void SetPathPrefix(const std::string& prefix) {
		path_prefix = prefix;
	}

	void SaveMidTensors(const std::string& path);
};