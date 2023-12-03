#pragma once
#include <iostream>

// Data split constants.
static const std::string TRAIN = "TRAIN";
static const std::string EVAL = "EVAL";

// TrainingConfig contains various configurations for a specific training run.
struct TrainingConfig {
    // Required flags with one arg (i.e. "--train=<file>")
    std::string train_file;
    std::string eval_file;

    // Optional flags with one arg and default values.
    uint32_t generate {0};
    uint32_t eval_interval {1000};
    uint32_t eval_iters {100};
    double learning_rate {1e-3};
    uint32_t epochs {0};
    uint32_t vocab_size {1000};
    std::string device {"cpu"};

    // Optional flags with one arg and no defaults.
    std::optional<std::string> tokenizer_model;
    std::optional<std::string> save_checkpoint;
    std::optional<std::string> load_checkpoint;
    std::optional<uint32_t> checkpoint_interval;

    // Optional flags with no args (e.g. "--verbose")
    bool verbose {false};
};