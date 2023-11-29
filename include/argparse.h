#include <iostream>
#include <string>
#include <functional>
// local includes
#include "tokenizer.h"
#include "train.h"

// NoArgHandler is a function which accepts a reference to a TrainingConfig and
// a flag with no arguments (i.e. "--verbose"). The function is intended to
// update the TrainingConfig in-place based on the flag passed.
using NoArgHandler = std::function<void(TrainingConfig&)>;

// OneArgHandler is a function which accepts a reference to a TrainingConfig and
// a flag with a single argument (i.e. --key value). The function is intended to
// update the TrainingConfig in-place based on the argument name and value passed.
using OneArgHandler = std::function<void(TrainingConfig&, const std::string& arg)>;

// Flags with no arguments.
#define NO_ARG(str, f, v) {str, [](TrainingConfig& cfg) {cfg.f = v;}}
const std::unordered_map<std::string, NoArgHandler> NoArgs {
        NO_ARG("--verbose", verbose, true),
};
#undef NO_ARG

// Flags with one argument.
#define ONE_ARG(str, f, v) {str, [](TrainingConfig& cfg, const std::string& arg) {cfg.f = v;}}
const std::unordered_map<std::string, OneArgHandler> OneArgs {
        ONE_ARG("--train", train_file, arg),
        ONE_ARG("--eval", eval_file, arg),
        ONE_ARG("--epochs", epochs, stoi(arg)),
        ONE_ARG("--tokenizer", tokenizer_model, arg),
        ONE_ARG("--save-checkpoint", save_checkpoint, arg),
        ONE_ARG("--load-checkpoint", load_checkpoint, arg),
        ONE_ARG("--checkpoint-interval", checkpoint_interval, stoi(arg)),
        ONE_ARG("--eval-interval", eval_interval, stoi(arg)),
        ONE_ARG("--eval-iters", eval_iters, stoi(arg)),
        ONE_ARG("--learning-rate", learning_rate, stof(arg)),
        ONE_ARG("--vocab-size", vocab_size, stoi(arg)),
        ONE_ARG("--generate", generate, stoi(arg)),
};
#undef ONE_ARG

// parse_args parses command line input into a TrainingConfig structure
// which contains the configurations for a single training run.
TrainingConfig& parse_args(int argc, char* argv[]);