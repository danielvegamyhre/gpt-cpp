#pragma once
#include <train.h>

namespace tokenizer {
    int train(const TrainingConfig& cfg, const unsigned int &vocab_size);
}