import os
import PIL.Image
import torch
import numpy as np
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
import time
import re
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# Specify the path to the model
model_path = r'D:\CodeRepo\VisualStudioSource\Janus.Cpp\Janus.Cpp\Janus-Pro-7B'
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(
    model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
).eval()

##################################################
output_dir = r'' # YOU NEED TO CHANGE THIS VARIABLE
##################################################

vl_gpt.language_model.model.norm.weight.data.numpy().view(
    dtype=np.uint8).tofile(f"{output_dir}/norm.weight.bin")

for i, layer in enumerate(vl_gpt.language_model.model.layers):
    layer.input_layernorm.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.input_layernorm.weight.bin")
    layer.post_attention_layernorm.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.post_attention_layernorm.weight.bin")

vl_gpt.gen_embed.weight.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/align_embeddings.bin")

vl_gpt.language_model.model.embed_tokens.weight.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/embed_tokens.bin")

vl_gpt = vl_gpt.to(torch.float16)

for i, layer in enumerate(vl_gpt.language_model.model.layers):
    layer.self_attn.k_proj.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.self_attn.k_proj.weight.bin")
    layer.self_attn.v_proj.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.self_attn.v_proj.weight.bin")
    layer.self_attn.q_proj.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.self_attn.q_proj.weight.bin")
    layer.self_attn.o_proj.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.self_attn.o_proj.weight.bin")
    layer.mlp.gate_proj.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.mlp.gate_proj.weight.bin")
    layer.mlp.up_proj.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.mlp.up_proj.weight.bin")
    layer.mlp.down_proj.weight.data.numpy().view(
        dtype=np.uint8).tofile(f"{output_dir}/layers.{i}.mlp.down_proj.weight.bin")

vl_gpt.gen_head.output_mlp_projector.weight.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/output_mlp_projector.bin")
vl_gpt.gen_head.vision_head.weight.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/vision_head.bin")

vl_gpt.gen_aligner.layers[0].weight.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/mlp_p1.bin")
vl_gpt.gen_aligner.layers[0].bias.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/mlp_p1_bias.bin")

vl_gpt.gen_aligner.layers[2].weight.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/mlp_p2.bin")
vl_gpt.gen_aligner.layers[2].bias.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/mlp_p2_bias.bin")