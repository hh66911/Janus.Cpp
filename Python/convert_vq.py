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
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.float16).eval()

##################################################
output_dir = r'' # YOU NEED TO CHANGE THIS VARIABLE
##################################################

def save_res_block(module, name):
    module.conv1.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.conv_in.weight.bin")
    module.conv1.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.conv_in.bias.bin")
    module.conv2.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.conv_out.weight.bin")
    module.conv2.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.conv_out.bias.bin")
    module.norm1.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.norm1.weight.bin")
    module.norm1.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.norm1.bias.bin")
    module.norm2.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.norm2.weight.bin")
    module.norm2.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.norm2.bias.bin")
    if hasattr(module, 'nin_shortcut'):
        module.nin_shortcut.weight.data.numpy().view(
            np.uint8).tofile(f"{name}.nin_shortcut.weight.bin")
        module.nin_shortcut.bias.data.numpy().view(
            np.uint8).tofile(f"{name}.nin_shortcut.bias.bin")
    
def save_attn_block(module, name):
    module.norm.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.norm.weight.bin")
    module.norm.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.norm.bias.bin")
    module.q.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.q_proj.weight.bin")
    module.q.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.q_proj.bias.bin")
    module.k.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.k_proj.weight.bin")
    module.k.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.k_proj.bias.bin")
    module.v.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.v_proj.weight.bin")
    module.v.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.v_proj.bias.bin")
    module.proj_out.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.o_proj.weight.bin")
    module.proj_out.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.o_proj.bias.bin")
    
def save_mid_block(module, name):
    save_res_block(module[0], f"{name}.res1")
    save_attn_block(module[1], f"{name}.attn")
    save_res_block(module[2], f"{name}.res2")
    
def save_upsample_block(module, name):
    module.conv.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.conv.weight.bin")
    module.conv.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.conv.bias.bin")
    
def save_one_conv_block(module, name):
    if hasattr(module, 'res'):
        for i, res in enumerate(module.res):
            save_res_block(res, f"{name}.res.{i}")
    if hasattr(module, "attn"):
        for i, attn in enumerate(module.attn):
            save_attn_block(attn, f"{name}.attn.{i}")
    if hasattr(module, "upsample"):
        save_upsample_block(module.upsample, f"{name}.upsample")
    
def save_conv_blocks(module, name):
    for i, block in enumerate(module):
        save_one_conv_block(block, f"{name}.{i}")
    
def save_decoder(module, name):
    module.conv_in.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.conv_in.weight.bin")
    module.conv_in.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.conv_in.bias.bin")
    module.conv_out.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.conv_out.weight.bin")
    module.conv_out.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.conv_out.bias.bin")
    save_mid_block(module.mid, f"{name}.mid")
    save_conv_blocks(module.conv_blocks, f"{name}.conv_blocks")
    module.norm_out.weight.data.numpy().view(
        np.uint8).tofile(f"{name}.norm_out.weight.bin")
    module.norm_out.bias.data.numpy().view(
        np.uint8).tofile(f"{name}.norm_out.bias.bin")
    
save_decoder(vl_gpt.gen_vision_model.decoder, f"{output_dir}/vq/decoder")

vq_embs = vl_gpt.gen_vision_model.quantize.embedding.weight.data.numpy()
vq_embs = vq_embs / np.linalg.norm(vq_embs, axis=1, keepdims=True)
vq_embs.view(np.uint8).tofile(f"{output_dir}/vq/quantize.embedding.bin")

vl_gpt.gen_vision_model.quant_conv.weight.data.numpy().view(
np.uint8).tofile(f"{output_dir}/vq/quant_conv.weight.bin")
vl_gpt.gen_vision_model.quant_conv.bias.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/vq/quant_conv.bias.bin")
vl_gpt.gen_vision_model.post_quant_conv.weight.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/vq/post_quant_conv.weight.bin")
vl_gpt.gen_vision_model.post_quant_conv.bias.data.numpy().view(
    np.uint8).tofile(f"{output_dir}/vq/post_quant_conv.bias.bin")

vl_gpt.gen_aligner.layers[0].bias.half().detach().cpu().numpy().view(np.uint8).tofile(
    f"{output_dir}/mlp_p1_bias.bin")
vl_gpt.gen_aligner.layers[2].bias.half().detach().cpu().numpy().view(np.uint8).tofile(
    f"{output_dir}/mlp_p2_bias.bin")