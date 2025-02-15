# 介绍
[Janus](https://github.com/deepseek-ai/Janus) 是 DeepSeek 推出的一款统一多模态理解和生成模型。

Janus.Cpp 专注于 [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) 模型在低显存和量化后的文生图推理。

性能相对于 transformer 库使用 bf16 进行推理速度快 80%

# 安装
## 准备工作
- Visual Studio 2022 最新版 （或者其他支持 C++23 的编译器）
- CUDA > 12.0
## 依赖库（建议使用 [vcpkg](https://github.com/microsoft/vcpkg/) 安装）
- [OpenCV](https://github.com/opencv/opencv) @ latest
- [GGML](https://github.com/ggerganov/ggml) @ 475e01227333a3a29ed0859b477beabcc2de7b5e
