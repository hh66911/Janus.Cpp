#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>

#include "tokenizer_models.h"

BPEModel load_bpe_model(std::filesystem::path);

std::vector<int> tokenizer_encode(const BPEModel& model, std::vector<uint8_t> text);
std::string tokenizer_decode(const BPEModel& model, const std::vector<int>& tokens);