#include <iostream>
#include <string>
#include <functional>
// third party includes
#include <tokenizer.h>
// local includes
#include <train.h>

// NoArgHandler is a function which accepts a reference to a TrainingConfig and
// a flag with no arguments (i.e. "--verbose"). The function is intended to
// update the TrainingConfig in-place based on the flag passed.
typedef std::function<void(TrainingConfig&)> NoArgHandler;

// OneArgHandler is a function which accepts a reference to a TrainingConfig and
// a flag with a single argument (i.e. --key value). The function is intended to
// update the TrainingConfig in-place based on the argument name and value passed.
typedef std::function<void(TrainingConfig&, const std::string& arg)> OneArgHandler;

// Flags with no arguments.
#define NO_ARG(str, f, v) {str, [](TrainingConfig& cfg) {cfg.f = v;}}
const std::unordered_map<std::string, NoArgHandler> NoArgs {
    NO_ARG("--generate", generate, true),
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
};
#undef ONE_ARG

// parse_args parses command line input into a TrainingConfig structure
// which contains the configurations for a single training run.
TrainingConfig& parse_args(int argc, char* argv[]) {
    static TrainingConfig cfg;
    // argv[0] is the program name, so we start at index 1.
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];

        // Check if this is a no-arg flag.
        if (auto flag = NoArgs.find(opt); flag != NoArgs.end()) {
            flag->second(cfg);
        }

        // Check if this is a one-arg flag.
        else if (auto flag = OneArgs.find(opt); flag != OneArgs.end()) {
            // Check if the arg is actually specified.
            if (++i < argc) {
                flag->second(cfg, argv[i]);
            } else {
                throw std::runtime_error("missing param after: " + opt);
            }
        }

        // Throw runtime errors for unrecognized arguments.
        else {
            throw std::runtime_error("unrecognized flag: " + opt);
        }
    }
    return cfg;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " gpt_cpp --train <training data> --eval <eval data> \n[--epochs <epochs>] [--tokenizer <tokenizer model>] [--save-checkpoint <checkpoint file>] \n[--load-checkpoint <checkpoint file>] [--checkpoint-interval <interval>] \n[--eval-interval <interval>] [--eval-iters <iters>] [--learning-rate <alpha>]" << std::endl;
        return 1;
    }
    TrainingConfig& cfg = parse_args(argc, argv);
    tokenizer::train(cfg, 9000);
    return 0;
}
