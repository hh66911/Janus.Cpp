#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>

#include "tokenizer_models.h"

BPEModel load_bpe_model(std::filesystem::path);

std::vector<int> tokenizer_encode(const BPEModel& model, std::string text);
std::string tokenizer_decode(const BPEModel& model, const std::vector<int>& tokens);