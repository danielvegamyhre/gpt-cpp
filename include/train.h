#pragma once
#include <iostream>

// TrainingConfig contains various configurations for a specific training run.
struct TrainingConfig {
    // Required flags with one arg (i.e. "--train=<file>")
    std::string train_file;
    std::string eval_file;

    // Optional flags with one arg and default values.
    unsigned int generate {0};
    unsigned int eval_interval {1000};
    unsigned int eval_iters {100};
    double learning_rate {1e-3};
    unsigned int epochs {0};
    unsigned int vocab_size {1000};
    std::string device {"cpu"};

    // Optional flags with one arg and no defaults.
    std::optional<std::string> tokenizer_model;
    std::optional<std::string> save_checkpoint;
    std::optional<std::string> load_checkpoint;
    std::optional<unsigned int> checkpoint_interval;

    // Optional flags with no args (e.g. "--verbose")
    bool verbose {false};
};