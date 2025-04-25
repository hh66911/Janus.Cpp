# 介绍
[Janus](https://github.com/deepseek-ai/Janus) 是 DeepSeek 推出的一款统一多模态理解和生成模型。

Janus.Cpp 目前专注于 [Janus-Pro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B) 模型在低显存和量化后的文生图推理。

性能相对于 transformer 库使用 bf16 进行推理速度快 80%，使用 flash_attnetion 可以再提速 100%

实测：RTX 4060 Laptop 8GB 使用 Q8_0 量化的模型可以实现 112s 生成一张图（384x384）。

Janus.Cpp 现在支持基于 tcp/ip 的分布式推理，在显存不够的情况下可以通过两个节点组网凑够（ip 需能公网 ping 通）。

# 安装
## 准备工作
- Visual Studio 2022 最新版 （或者其他支持 C++23 的编译器）
- CUDA > 12.0
## 依赖库（建议使用 [vcpkg](https://github.com/microsoft/vcpkg/) 安装）
- [OpenCV](https://github.com/opencv/opencv) @ latest
- [GGML](https://github.com/ggerganov/ggml) @ latest
- [httplib](https://github.com/yhirose/cpp-httplib) @ latest

# 权重转换和量化
## 转换
Janus.Cpp 无法直接读取 .safetensors 文件，使用 `convert_vq.py` 和 `convert_lang.py` 文件提取出 .bin 权重到文件夹中，用户需要手动创建文件夹，并修改上述两个 Python 文件中的 `output_dir` 变量。

## 量化
Janus.CPP 可以从上一步的 .bin 文件生成量化的模型权重，

# TODO
- [x] Flash Attention
- [ ] 使用 GGUF
- [ ] 命令行 Argument Parser
- [ ] 多节点分布推理
- [ ] 支持更多回归生图、视频模型（MAGI-1, FramePack, ...）