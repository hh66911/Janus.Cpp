#pragma once

#include <ggml.h>
#include <iostream>
#include <ranges>

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
